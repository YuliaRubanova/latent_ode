###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter
from lib.base_models import Baseline, VAE_Baseline

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Exponential decay of the hidden states for RNN
# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/

# Exp decay between hidden states
class GRUCellExpDecay(RNNCellBase):
	def __init__(self, input_size, input_size_for_decay, hidden_size, device, bias=True):
		super(GRUCellExpDecay, self).__init__(input_size, hidden_size, bias, num_chunks=3)

		self.device = device
		self.input_size_for_decay = input_size_for_decay
		self.decay = nn.Sequential(nn.Linear(input_size_for_decay, 1),)
		utils.init_network_weights(self.decay)

	def gru_exp_decay_cell(self, input, hidden, w_ih, w_hh, b_ih, b_hh):
		# INPORTANT: assumes that cum delta t is the last dimension of the input
		batch_size, n_dims = input.size()
		
		# "input" contains the data, mask and also cumulative deltas for all inputs
		cum_delta_ts = input[:, -self.input_size_for_decay:]
		data = input[:, :-self.input_size_for_decay]

		decay = torch.exp( - torch.min(torch.max(
			torch.zeros([1]).to(self.device), self.decay(cum_delta_ts)), 
			torch.ones([1]).to(self.device) * 1000 ))

		hidden = hidden * decay

		gi = torch.mm(data, w_ih.t()) + b_ih
		gh = torch.mm(hidden, w_hh.t()) + b_hh
		i_r, i_i, i_n = gi.chunk(3, 1)
		h_r, h_i, h_n = gh.chunk(3, 1)

		resetgate = torch.sigmoid(i_r + h_r)
		inputgate = torch.sigmoid(i_i + h_i)
		newgate = torch.tanh(i_n + resetgate * h_n)
		hy = newgate + inputgate * (hidden - newgate)
		return hy

	def forward(self, input, hx=None):
		# type: (Tensor, Optional[Tensor]) -> Tensor
		#self.check_forward_input(input)
		if hx is None:
			hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
		#self.check_forward_hidden(input, hx, '')
		
		return self.gru_exp_decay_cell(
			input, hx,
			self.weight_ih, self.weight_hh,
			self.bias_ih, self.bias_hh
		)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Imputation with a weighed average of previous value and empirical mean 
# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
def get_cum_delta_ts(data, delta_ts, mask):
	n_traj, n_tp, n_dims = data.size()
	
	cum_delta_ts = delta_ts.repeat(1, 1, n_dims)
	missing_index = np.where(mask.cpu().numpy() == 0)

	for idx in range(missing_index[0].shape[0]):
		i = missing_index[0][idx] 
		j = missing_index[1][idx]
		k = missing_index[2][idx]

		if j != 0 and j != (n_tp-1):
		 	cum_delta_ts[i,j+1,k] = cum_delta_ts[i,j+1,k] + cum_delta_ts[i,j,k]
	cum_delta_ts = cum_delta_ts / cum_delta_ts.max() # normalize

	return cum_delta_ts


# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
# very slow
def impute_using_input_decay(data, delta_ts, mask, w_input_decay, b_input_decay):
	n_traj, n_tp, n_dims = data.size()

	cum_delta_ts = delta_ts.repeat(1, 1, n_dims)
	missing_index = np.where(mask.cpu().numpy() == 0)

	data_last_obsv = np.copy(data.cpu().numpy())
	for idx in range(missing_index[0].shape[0]):
		i = missing_index[0][idx] 
		j = missing_index[1][idx]
		k = missing_index[2][idx]

		if j != 0 and j != (n_tp-1):
		 	cum_delta_ts[i,j+1,k] = cum_delta_ts[i,j+1,k] + cum_delta_ts[i,j,k]
		if j != 0:
			data_last_obsv[i,j,k] = data_last_obsv[i,j-1,k] # last observation
	cum_delta_ts = cum_delta_ts / cum_delta_ts.max() # normalize
	
	data_last_obsv = torch.Tensor(data_last_obsv).to(get_device(data))

	zeros = torch.zeros([n_traj, n_tp, n_dims]).to(get_device(data))
	decay = torch.exp( - torch.min( torch.max(zeros, 
		w_input_decay * cum_delta_ts + b_input_decay), zeros + 1000 ))

	data_means = torch.mean(data, 1).unsqueeze(1)

	data_imputed = data * mask + (1-mask) * (decay * data_last_obsv + (1-decay) * data_means)
	return data_imputed


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def run_rnn(inputs, delta_ts, cell, first_hidden=None, 
	mask = None, feed_previous=False, n_steps=0,
	decoder = None, input_decay_params = None,
	feed_previous_w_prob = 0.,
	masked_update = True):
	if (feed_previous or feed_previous_w_prob) and decoder is None:
		raise Exception("feed_previous is set to True -- please specify RNN decoder")

	if n_steps == 0:
		n_steps = inputs.size(1)

	if (feed_previous or feed_previous_w_prob) and mask is None:
		mask = torch.ones((inputs.size(0), n_steps, inputs.size(-1))).to(get_device(inputs))

	if isinstance(cell, GRUCellExpDecay):
		cum_delta_ts = get_cum_delta_ts(inputs, delta_ts, mask)

	if input_decay_params is not None:
		w_input_decay, b_input_decay = input_decay_params
		inputs = impute_using_input_decay(inputs, delta_ts, mask,
			w_input_decay, b_input_decay)

	all_hiddens = []
	hidden = first_hidden

	if hidden is not None:
		all_hiddens.append(hidden)
		n_steps -= 1

	for i in range(n_steps):
		delta_t = delta_ts[:,i]
		if i == 0:
			rnn_input = inputs[:,i]
		elif feed_previous:
			rnn_input = decoder(hidden)
		elif feed_previous_w_prob > 0:
			feed_prev = np.random.uniform() > feed_previous_w_prob
			if feed_prev:
				rnn_input = decoder(hidden)
			else:
				rnn_input = inputs[:,i]
		else:
			rnn_input = inputs[:,i]

		if mask is not None:
			mask_i = mask[:,i,:]
			rnn_input = torch.cat((rnn_input, mask_i), -1)

		if isinstance(cell, GRUCellExpDecay):
			cum_delta_t = cum_delta_ts[:,i]
			input_w_t = torch.cat((rnn_input, cum_delta_t), -1).squeeze(1)
		else:
			input_w_t = torch.cat((rnn_input, delta_t), -1).squeeze(1)

		prev_hidden = hidden
		hidden = cell(input_w_t, hidden)

		if masked_update and (mask is not None) and (prev_hidden is not None):
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			summed_mask = (torch.sum(mask_i, -1, keepdim = True) > 0).float()
			assert(not torch.isnan(summed_mask).any())
			hidden = summed_mask * hidden + (1-summed_mask) * prev_hidden

		all_hiddens.append(hidden)

	all_hiddens = torch.stack(all_hiddens, 0)
	all_hiddens = all_hiddens.permute(1,0,2).unsqueeze(0)
	return hidden, all_hiddens




class Classic_RNN(Baseline):
	def __init__(self, input_dim, latent_dim, device, 
		concat_mask = False, obsrv_std = 0.1, 
		use_binary_classif = False,
		linear_classifier = False,
		classif_per_tp = False,
		input_space_decay = False,
		cell = "gru", n_units = 100,
		n_labels = 1,
		train_classif_w_reconstr = False):
		
		super(Classic_RNN, self).__init__(input_dim, latent_dim, device, 
			obsrv_std = obsrv_std, 
			use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = linear_classifier,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		self.concat_mask = concat_mask
		
		encoder_dim = int(input_dim)
		if concat_mask:
			encoder_dim = encoder_dim * 2

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim),)

		#utils.init_network_weights(self.encoder)
		utils.init_network_weights(self.decoder)

		if cell == "gru":
			self.rnn_cell = GRUCell(encoder_dim + 1, latent_dim) # +1 for delta t
		elif cell == "expdecay":
			self.rnn_cell = GRUCellExpDecay(
				input_size = encoder_dim, 
				input_size_for_decay = input_dim,
				hidden_size = latent_dim, 
				device = device)
		else:
			raise Exception("Unknown RNN cell: {}".format(cell))

		if input_space_decay:
			self.w_input_decay =  Parameter(torch.Tensor(1, int(input_dim))).to(self.device)
			self.b_input_decay =  Parameter(torch.Tensor(1, int(input_dim))).to(self.device)
		self.input_space_decay = input_space_decay

		self.z0_net = lambda hidden_state: hidden_state


	def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
		mask = None, n_traj_samples = 1, mode = None):

		assert(mask is not None)
		n_traj, n_tp, n_dims = data.size()

		if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
			raise Exception("Extrapolation mode not implemented for RNN models")

		# for classic RNN time_steps_to_predict should be the same as  truth_time_steps
		assert(len(truth_time_steps) == len(time_steps_to_predict))

		batch_size = data.size(0)
		zero_delta_t = torch.Tensor([0.]).to(self.device)

		delta_ts = truth_time_steps[1:] - truth_time_steps[:-1]
		delta_ts = torch.cat((delta_ts, zero_delta_t))
		if len(delta_ts.size()) == 1:
			# delta_ts are shared for all trajectories in a batch
			assert(data.size(1) == delta_ts.size(0))
			delta_ts = delta_ts.unsqueeze(-1).repeat((batch_size,1,1))

		input_decay_params = None
		if self.input_space_decay:
			input_decay_params = (self.w_input_decay, self.b_input_decay)

		if mask is not None:
			utils.check_mask(data, mask)

		hidden_state, all_hiddens = run_rnn(data, delta_ts, 
			cell = self.rnn_cell, mask = mask,
			input_decay_params = input_decay_params,
			feed_previous_w_prob = (0. if self.use_binary_classif else 0.5),
			decoder = self.decoder)

		outputs = self.decoder(all_hiddens)
		# Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
		first_point = data[:,0,:]
		outputs = utils.shift_outputs(outputs, first_point)

		extra_info = {"first_point": (hidden_state.unsqueeze(0), 0.0, hidden_state.unsqueeze(0))}

		if self.use_binary_classif:
			if self.classif_per_tp:
				extra_info["label_predictions"] = self.classifier(all_hiddens)
			else:
				extra_info["label_predictions"] = self.classifier(hidden_state).reshape(1,-1)

		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		return outputs, extra_info



class RNN_VAE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, rec_dims, 
		z0_prior, device, 
		concat_mask = False, obsrv_std = 0.1, 
		input_space_decay = False,
		use_binary_classif = False,
		classif_per_tp =False,
		linear_classifier = False, 
		cell = "gru", n_units = 100,
		n_labels = 1,
		train_classif_w_reconstr = False):
	
		super(RNN_VAE, self).__init__(
			input_dim = input_dim, latent_dim = latent_dim, 
			z0_prior = z0_prior, 
			device = device, obsrv_std = obsrv_std, 
			use_binary_classif = use_binary_classif, 
			classif_per_tp = classif_per_tp,
			linear_classifier = linear_classifier,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		self.concat_mask = concat_mask

		encoder_dim = int(input_dim)
		if concat_mask:
			encoder_dim = encoder_dim * 2

		if cell == "gru":
			self.rnn_cell_enc = GRUCell(encoder_dim + 1, rec_dims) # +1 for delta t
			self.rnn_cell_dec = GRUCell(encoder_dim + 1, latent_dim) # +1 for delta t
		elif cell == "expdecay":
			self.rnn_cell_enc = GRUCellExpDecay(
				input_size = encoder_dim,
				input_size_for_decay = input_dim,
				hidden_size = rec_dims, 
				device = device)
			self.rnn_cell_dec = GRUCellExpDecay(
				input_size = encoder_dim,
				input_size_for_decay = input_dim,
				hidden_size = latent_dim, 
				device = device) 
		else:
			raise Exception("Unknown RNN cell: {}".format(cell))

		self.z0_net = nn.Sequential(
			nn.Linear(rec_dims, n_units),
			nn.Tanh(),
			nn.Linear(n_units, latent_dim * 2),)
		utils.init_network_weights(self.z0_net)

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim),)

		#utils.init_network_weights(self.encoder)
		utils.init_network_weights(self.decoder)

		if input_space_decay:
			self.w_input_decay =  Parameter(torch.Tensor(1, int(input_dim))).to(self.device)
			self.b_input_decay =  Parameter(torch.Tensor(1, int(input_dim))).to(self.device)
		self.input_space_decay = input_space_decay

	def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
		mask = None, n_traj_samples = 1, mode = None):

		assert(mask is not None)

		batch_size = data.size(0)
		zero_delta_t = torch.Tensor([0.]).to(self.device)
	
		# run encoder backwards
		run_backwards = bool(time_steps_to_predict[0] < truth_time_steps[-1])

		if run_backwards:
			# Look at data in the reverse order: from later points to the first
			data = utils.reverse(data)
			mask = utils.reverse(mask)

		delta_ts = truth_time_steps[1:] - truth_time_steps[:-1]
		if run_backwards:
			# we are going backwards in time
			delta_ts = utils.reverse(delta_ts)


		delta_ts = torch.cat((delta_ts, zero_delta_t))
		if len(delta_ts.size()) == 1:
			# delta_ts are shared for all trajectories in a batch
			assert(data.size(1) == delta_ts.size(0))
			delta_ts = delta_ts.unsqueeze(-1).repeat((batch_size,1,1))

		input_decay_params = None
		if self.input_space_decay:
			input_decay_params = (self.w_input_decay, self.b_input_decay)

		hidden_state, _ = run_rnn(data, delta_ts, 
			cell = self.rnn_cell_enc, mask = mask,
			input_decay_params = input_decay_params)

		z0_mean, z0_std = utils.split_last_dim(self.z0_net(hidden_state))
		z0_std = z0_std.abs()
		z0_sample = utils.sample_standard_gaussian(z0_mean, z0_std)

		# Decoder # # # # # # # # # # # # # # # # # # # #
		delta_ts = torch.cat((zero_delta_t, time_steps_to_predict[1:] - time_steps_to_predict[:-1]))
		if len(delta_ts.size()) == 1:
			delta_ts = delta_ts.unsqueeze(-1).repeat((batch_size,1,1))

		_, all_hiddens = run_rnn(data, delta_ts,
			cell = self.rnn_cell_dec,
			first_hidden = z0_sample, feed_previous = True, 
			n_steps = time_steps_to_predict.size(0),
			decoder = self.decoder,
			input_decay_params = input_decay_params)

		outputs = self.decoder(all_hiddens)
		# Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
		first_point = data[:,0,:]
		outputs = utils.shift_outputs(outputs, first_point)

		extra_info = {"first_point": (z0_mean.unsqueeze(0), z0_std.unsqueeze(0), z0_sample.unsqueeze(0))}

		if self.use_binary_classif:
			if self.classif_per_tp:
				extra_info["label_predictions"] = self.classifier(all_hiddens)
			else:
				extra_info["label_predictions"] = self.classifier(z0_mean).reshape(1,-1)

		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		return outputs, extra_info



