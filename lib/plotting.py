import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import os
from scipy.stats import kde

import numpy as np
import subprocess
import torch
import lib.utils as utils
import matplotlib.gridspec as gridspec
from lib.utils import get_device

from lib.encoder_decoder import *
from lib.rnn_baselines import *
from lib.ode_rnn import *
import torch.nn.functional as functional
from torch.distributions.normal import Normal
from lib.odevae import ODEVAE

from lib.likelihood_eval import masked_gaussian_log_density
try:
	import umap
except:
	print("Couldn't import umap")

from generate_timeseries import Periodic_1d
from person_activity import PersonActivity

from lib.utils import compute_loss_all_batches



def plot_trajectories(ax, traj, time_steps, min_y = None, max_y = None, title = "", 
		add_to_plot = False, label = None, add_legend = False, dim_to_show = 0,
		linestyle = '-', marker = 'o', mask = None):
	# expected shape of traj: [n_traj, n_timesteps, n_dims]
	# The function will produce one line per trajectory (n_traj lines in total)
	if not add_to_plot:
		ax.cla()
	ax.set_title(title)
	ax.set_xlabel('Time')
	ax.set_ylabel('x')
	
	if min_y is not None:
		ax.set_ylim(bottom = min_y)

	if max_y is not None:
		ax.set_ylim(top = max_y)

	for i in range(traj.size()[0]):
		if len(time_steps.size()) == 1:
			time_steps_one = time_steps
		else:
			time_steps_one = time_steps[:,i]

		d = traj[i].cpu().numpy()[:, dim_to_show]
		ts = time_steps_one.cpu().numpy()
		if mask is not None:
			m = mask[i].cpu().numpy()[:, dim_to_show]
			d = d[m == 1]
			ts = ts[m == 1]
		ax.plot(ts, d, linestyle = linestyle, label = label, marker=marker)

	if add_legend:
		ax.legend()


def plot_std(ax, traj, traj_std, time_steps, min_y = None, max_y = None, title = "", 
	add_to_plot = False, label = None, alpha=0.2):

	# take only the first (and only?) dimension
	mean_minus_std = (traj - traj_std).cpu().numpy()[:, :, 0]
	mean_plus_std = (traj + traj_std).cpu().numpy()[:, :, 0]

	for i in range(traj.size()[0]):
		if len(time_steps.size()) == 1:
			time_steps_one = time_steps
		else:
			time_steps_one = time_steps[:,i]

		ax.fill_between(time_steps_one.cpu().numpy(), mean_minus_std[i], mean_plus_std[i], alpha=alpha)



class Visualizations():
	def __init__(self, device):
		self.init_visualization()
		self.device = device

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 6), facecolor='white')
		
		self.ax_true_traj = self.fig.add_subplot(2,2,1, frameon=False)
		#self.ax_gen_traj = self.fig.add_subplot(2,2,2, frameon=False)

		self.ax_y0 = self.fig.add_subplot(2,2,2, frameon=False)	

		self.ax_samples_same_traj = self.fig.add_subplot(2,2,3, frameon=False)
		self.ax_traj_from_prior = self.fig.add_subplot(2,2,4, frameon=False)	

		plt.show(block=False)


	def draw_all_plots_one_dim(self, data_dict, model,
		loss_list = None, kl_list = None, gp_param_list = None, y0_gp_param_list = None,
		call_count_d  = None, plot_name = "", save = False, experimentID = 0.):

		data =  data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]
		
		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		device = get_device(time_steps)

		time_steps_to_predict = time_steps
		if isinstance(model, ODEVAE):
			# sample at the original time points
			time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 50).to(device)

		reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
			observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 10)

		n_traj_to_show = 5
		# plot only 10 trajectories
		data_for_plotting = data[:n_traj_to_show]
		reconstructions_for_plotting = reconstructions[0,:n_traj_to_show]
		reconstr_std = reconstructions.std(dim=0)[:n_traj_to_show]

		dim_to_show = 0
		max_y = max(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().max(),
			reconstructions[:,:,dim_to_show].cpu().numpy().max())
		min_y = min(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().min(),
			reconstructions[:,:,dim_to_show].cpu().numpy().min())

		############################################
		# Plot truth and reconstruction

		plot_trajectories(self.ax_true_traj, data_for_plotting, time_steps, 
			min_y = min_y, max_y = max_y, title="True trajectories", 
			marker = 'o', linestyle='dashed', dim_to_show = dim_to_show)
		plot_trajectories(self.ax_true_traj, reconstructions_for_plotting, time_steps_to_predict, 
			min_y = min_y, max_y = max_y, title="Reconstructed trajectories", dim_to_show = dim_to_show,
			add_to_plot = True)
		plot_std(self.ax_true_traj, reconstructions_for_plotting, reconstr_std, time_steps_to_predict, alpha=0.5)

		############################################
		# Plot loss and KL
		if loss_list is not None: 
			self.ax_loss.cla()
			for key, val_list in loss_list.items():
				self.ax_loss.plot(val_list, label = key)
				self.ax_loss.set_title("Loss")
				self.ax_loss.legend()

		if kl_list is not None:
			self.ax_kl.plot(kl_list, color = "r")
			self.ax_kl.set_title("KL div for y0")

		############################################
		# Get several samples for the same trajectory
		gt_traj = data_for_plotting[:1]
		first_point = gt_traj[:,0]

		if len(time_steps_to_predict.size()) == 1:
			time_steps_to_predict_one = time_steps_to_predict
			time_steps_one = time_steps
		else:
			time_steps_to_predict_one = time_steps_to_predict[:,0]
			time_steps_one = time_steps[:,0]

		samples_same_traj, _ = model.get_reconstruction(time_steps_to_predict_one, 
			observed_data[:1], observed_time_steps, mask = observed_mask[:1], n_traj_samples = 5)
		samples_same_traj = samples_same_traj.squeeze(1)
		
		plot_trajectories(self.ax_samples_same_traj, samples_same_traj, time_steps_to_predict_one)
		plot_trajectories(self.ax_samples_same_traj, gt_traj, time_steps_one, linestyle = "dashed", 
			label = "True traj", add_to_plot = True, title="Samples for the same trajectory")

		############################################
		# Plot trajectories from prior
		
		if isinstance(model, ODEVAE):
			traj_from_prior = model.sample_traj_from_prior(time_steps_to_predict_one, n_traj_samples = 6)
			# Since in this case n_traj = 1, n_traj_samples -- requested number of samples from the prior, squeeze n_traj dimension
			traj_from_prior = traj_from_prior.squeeze(1)

			plot_trajectories(self.ax_traj_from_prior, traj_from_prior, time_steps_to_predict_one, title="Trajectories from prior")

		################################################

		# Plot y0
		first_point_mu, first_point_std, first_point_enc = info["first_point"]

		dim1 = 0
		dim2 = 1
		self.ax_y0.cla()
		# first_point_enc shape: [1, n_traj, n_dims]
		self.ax_y0.scatter(first_point_enc.cpu()[0,:,dim1], first_point_enc.cpu()[0,:,dim2])
		self.ax_y0.set_title("First points y0 for all test trajectories")
		self.ax_y0.set_xlabel('dim {}'.format(dim1))
		self.ax_y0.set_ylabel('dim {}'.format(dim2))

		################################################

		self.fig.tight_layout()
		plt.draw()

		if save:
			dirname = "plots/" + str(experimentID) + "/"
			os.makedirs(dirname, exist_ok=True)
			self.fig.savefig(dirname + plot_name + "_" + str(experimentID) + "_fig.pdf")









our_green = '#2ECC40'
our_green2 = '#3D9970'
our_red = '#FF4136'
our_blue = '#0074D9'

def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
LARGE_SIZE = 22

def init_fonts(main_font_size = LARGE_SIZE):
	plt.rc('font', size=main_font_size)          # controls default text sizes
	plt.rc('axes', titlesize=main_font_size)     # fontsize of the axes title
	plt.rc('axes', labelsize=main_font_size)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=main_font_size)  # fontsize of the figure title



def plot_kde_density(ax, data, time_steps, min_y = None, max_y = None, title = "Density (gaussian KDE)"):
	ax.cla()
	ax.set_title(title)
	ax.set_xlabel('t')
	ax.set_ylabel('value')

	num_samples, n_timepoints, dim = data.shape
	assert(n_timepoints == time_steps.size()[0])

	time_tiled = np.expand_dims(np.tile(time_steps.cpu().numpy(), (num_samples,1)),2)
	concat = np.concatenate((time_tiled, data), 2).reshape(-1, 2)
	x, y = concat.T

	# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
	nbins = 50

	s = time.time()
	k = kde.gaussian_kde(concat.T)

	if min_y is None:
		min_y = y.min()

	if max_y is None:
		max_y = y.max()

	xi, yi = np.mgrid[x.min():x.max():nbins*1j, min_y:max_y:nbins*1j]
	stacked = np.vstack([xi.flatten(), yi.flatten()])

	zi = k(stacked)
	
	# plot a density
	ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)


def logsumexp(inputs):
	return (inputs - functional.log_softmax(inputs)).mean()


def plot_estim_density_samples_q(ax, log_density_func, reconstructions, time_steps, 
	min_y = None, max_y = None, title = "Sample density (MC estimate)",
	dim_to_show = 0):

	ax.cla()
	ax.set_title(title)
	ax.set_xlabel('t')
	ax.set_ylabel('value')

	# make sample on regular grid!!!!!

	# Compute E_{q(z| ground_truth)} [p(grid|z)]
	n_samples, n_timepoints, dim = reconstructions.size()

	npts = 70

	if min_y is None:
		min_y = reconstructions.min()

	if max_y is None:
		max_y = reconstructions.max()

	y_grid = torch.from_numpy(np.expand_dims(np.linspace(min_y, max_y, npts),1)).type(torch.float32)
	y_grid = y_grid.view(1,-1,1,1)

	density_grid = []
	for t_index in range(n_timepoints):
		samples_t = reconstructions[:,t_index:t_index+1, dim_to_show:(dim_to_show+1)].unsqueeze(1)

		# shape [n_grid_points, n_traj_samples]
		p_grid_given_z_current_t = log_density_func(samples_t, y_grid)
		p_grid_given_z_current_t = np.exp(p_grid_given_z_current_t)
		density = torch.mean(p_grid_given_z_current_t,1)
		density_grid.append(density)

	density_grid = torch.stack(density_grid)
	density_grid = density_grid.cpu().numpy().transpose()

	xx, yy = np.meshgrid(time_steps, y_grid)
	ax.pcolormesh(xx, yy, density_grid, shading='gouraud', cmap=plt.cm.BuGn_r)

	# plot the samples on top of the density
	# for i in range(reconstructions.size()[0]):
	#   ax.plot(time_steps.cpu().numpy(), reconstructions[i].cpu().numpy()[:, 0], '-')




def plot_estim_density_samples_prior(ax, model, time_steps, 
	min_y = None, max_y = None, title = "Sample density (MC estimate)",
	dim_to_show = 0):

	ax.cla()
	ax.set_title(title)
	ax.set_xlabel('t')
	ax.set_ylabel('value')
	
	# Compute E_{p(z)} [p(grid|z)]

	time_steps_regular_grid = torch.linspace(time_steps.min(), time_steps.max(), 100)

	# sample z from prior, decode into y
	traj_from_prior = model.sample_traj_from_prior(time_steps_regular_grid, n_traj_samples = 100)
	# n_traj = 1
	n_traj_samples, n_traj, n_timepoints, dim = traj_from_prior.size()

	npts = 70

	if min_y is None:
		min_y = traj_from_prior.min()

	if max_y is None:
		max_y = traj_from_prior.max()

	y_grid = torch.from_numpy(np.linspace(min_y, max_y, npts)).type(torch.float32)
	if traj_from_prior.is_cuda:
		y_grid = y_grid.to(traj_from_prior.get_device())
	y_grid = y_grid.view(1,-1,1,1)

	# two for loops to make backward-compatibility with other scripts -- needs to be re-written for faster implementation
	density_grid = []
	for t_index in range(n_timepoints):
		traj_from_prior_t = traj_from_prior[:,:,t_index:t_index+1, dim_to_show:(dim_to_show+1)]

		# shape [n_grid_points, n_traj_samples]
		p_grid_given_z_current_t = model.masked_gaussian_log_density(traj_from_prior_t, y_grid)
		p_grid_given_z_current_t = torch.exp(p_grid_given_z_current_t)

		# take mean of GP samples
		density = torch.mean(p_grid_given_z_current_t,1)
		density_grid.append(density)
	
	# stack over time points
	density_grid = torch.stack(density_grid)
	density_grid = density_grid.cpu().numpy().transpose()

	y_grid = y_grid.cpu().numpy()
	xx, yy = np.meshgrid(time_steps_regular_grid, y_grid)
	ax.pcolormesh(xx, yy, density_grid, shading='gouraud', cmap=plt.cm.BuGn_r)


def plot_estim_density_samples_prior_conditioned_on_data(ax, model, gt_data, gt_time_steps, min_y = None, max_y = None, title = "Sample density (MC estimate)"):
	ax.cla()
	ax.set_title(title)
	ax.set_xlabel('t')
	ax.set_ylabel('value')
	
	# Compute E_{p(z| ground truth)}[p(grid|z)] = E_{p(z)} [p(grid|z) prod_t p(x_t | z)], where prod_t p(x_t | z) -- product over all time points of one trajectory
	# we want to see if this will look like a GP with increasing variance between the points

	# Take one trajectory from truth
	traj = gt_data[:1].unsqueeze(0)

	# Take samples on the regular grid to show how the variance changes between ground-truth points
	time_steps_regular_grid = torch.linspace(gt_time_steps.min()-0.01, gt_time_steps.max()+0.01, 50)
	# Concatenate with time_steps used for observed data so that we can evaluate likelihood of gt_data under sample z
	time_steps_cat = torch.cat((time_steps_regular_grid, gt_time_steps))
	sort_order = np.argsort(np.argsort(time_steps_cat).cpu().numpy())
	gt_time_points_idx = sort_order[len(time_steps_regular_grid):]
	time_steps_cat = torch.sort(time_steps_cat)[0]

	# sample z from prior, decode into y
	traj_from_prior = model.sample_traj_from_prior(time_steps_cat, n_traj_samples = 100)
	# n_traj = 1
	n_traj_samples, n_traj, n_timepoints, dim = traj_from_prior.size()

	# shape: [n_traj, n_traj_samples]; n_traj = 1
	p_gt_data_given_z_t = model.masked_gaussian_log_density(traj_from_prior[:,:,gt_time_points_idx], traj)
	p_gt_data_given_z_t = p_gt_data_given_z_t - torch.mean(p_gt_data_given_z_t,1)

	npts = 70

	if min_y is None:
		min_y = traj_from_prior.min()

	if max_y is None:
		max_y = traj_from_prior.max()

	y_grid = torch.from_numpy(np.linspace(min_y, max_y, npts)).type(torch.float32)
	y_grid = y_grid.view(1,-1,1,1)

	# two for loops to make backward-compatibility with other scripts -- needs to be re-written for faster implementation
	density_grid = []
	for t_index in range(n_timepoints):
		traj_from_prior_t = traj_from_prior[:,:,t_index:t_index+1]

		# shape [n_grid_points, n_traj_samples]
		p_grid_given_z_current_t = model.masked_gaussian_log_density(traj_from_prior_t, y_grid)
		prod = p_grid_given_z_current_t + p_gt_data_given_z_t
		prod = np.exp(prod)

		# take mean of GP samples
		density = torch.mean(prod,1)
		density_grid.append(density)
	
	# stack over time points
	density_grid = torch.stack(density_grid)
	density_grid = density_grid.cpu().numpy().transpose()

	xx, yy = np.meshgrid(time_steps_cat, y_grid)
	ax.pcolormesh(xx, yy, density_grid, shading='gouraud', cmap=plt.cm.BuGn_r)


def convert_to_movie(img_name, video_name, rate = 10):
	# Combine the images into a movie
    bashCommand = r"ffmpeg -r {} -y -i {} -r 10 {}".format(int(rate), img_name, video_name)
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,
    	shell=False, stderr=subprocess.PIPE)
    output, error = process.communicate()



def make_conditioning_on_ind_points_video(model, truth, truth_time_steps, experimentID):
	one_traj = truth[1:2]
	first_point = one_traj[:,0]
	n_traj_samples = 10
	n_timepoints_to_condition = len(truth_time_steps)

	mixtures = None
	# if model.n_ode_mixtures is not None:
	# 	mixtures = model.mixture_nn(first_point_enc)
	# 	mixtures = mixtures / mixtures.sum(-1, keepdim = True)

	int_length = truth_time_steps.max() - truth_time_steps.min()
	time_steps_to_predict = torch.linspace(truth_time_steps.min(), truth_time_steps.max() + int_length/2, 100)
	time_steps_to_predict = time_steps_to_predict.to(get_device(truth))

	fig = plt.figure(figsize=(8, 4), facecolor='white')
	ax = fig.add_subplot(111, frameon=False)

	p = np.random.permutation(len(truth_time_steps))

	for i in range(1, n_timepoints_to_condition):
		sort_order = np.argsort(p[:i])
		idx_sorted = p[:i][sort_order]

		pred_x, info = model.get_reconstruction(time_steps_to_predict, 
			one_traj[:,idx_sorted], truth_time_steps[idx_sorted], 
			mask = torch.ones_like(one_traj[:,idx_sorted]).to(get_device(truth)), 
			n_traj_samples = n_traj_samples)

		ax.cla()
		plot_trajectories(ax, pred_x.squeeze(1), time_steps_to_predict, title="", marker = None, linestyle = '-')
		plot_trajectories(ax, one_traj[:,idx_sorted], truth_time_steps[idx_sorted], 
			title="", marker = 'o', linestyle = '', add_to_plot = True)
		ax.set_ylim(one_traj.cpu().min()-0.2, one_traj.cpu().max() + 0.5)

		#ax.axvline(x=0.)

		dirname = "plots/cond_on_ind_points/" + str(experimentID) + "/"
		os.makedirs(dirname, exist_ok=True)

		fig.savefig(dirname + "cond_on_ind_points_" + str(experimentID) + "_{:03d}".format(i) + ".png")
		plt.close()

	convert_to_movie(dirname + "/cond_on_ind_points_" + str(experimentID) + "_%03d.png", 
		"plots/" + str(experimentID) + "/conditioning_on_ind_points_" + str(experimentID) + ".mp4",
		rate = 1)


# def save_reconstructions_for_same_traj(model, truth, truth_time_steps, itr, experimentID):
# 	one_traj = truth[1:2]

# 	time_steps_to_predict = utils.linspace_vector(truth_time_steps[0], truth_time_steps[-1]*1.5, 100)

# 	print("save_reconstructions_for_same_traj")
# 	print(truth_time_steps[:,:10])
# 	print(time_steps_to_predict[:,:10])
# 	print(truth_time_steps.size())
# 	print(time_steps_to_predict.size())


# 	samples_same_traj, _ = model.get_reconstruction(time_steps_to_predict, one_traj, truth_time_steps, n_traj_samples = 10)
# 	samples_same_traj = samples_same_traj.squeeze(1)
	
# 	fig = plt.figure(figsize=(8, 4), facecolor='white')
# 	ax = fig.add_subplot(111, frameon=False)

# 	plot_trajectories(ax, samples_same_traj, time_steps_to_predict, marker = None, linestyle = '-')
# 	plot_trajectories(ax, one_traj, truth_time_steps, linestyle = "dashed", 
# 		label = "True traj", add_to_plot = True, title="Samples for the same trajectory")
# 	ax.set_ylim(one_traj.cpu().min()-0.2, one_traj.cpu().max() + 0.5)

# 	dirname = "plots/samples_same_traj/" + str(experimentID) + "/"
# 	os.makedirs(dirname, exist_ok=True)

# 	fig.savefig(dirname + "samples_same_traj_" + str(experimentID) + "_{:04d}".format(itr) + ".png")
# 	plt.close()

########################################################################################
def save_all_dims(plot_func, plot_file_name, title, n_dims):
	fig = plt.figure(figsize=(6*n_dims, 4), facecolor='white')
	for d in range(n_dims):
		ax = fig.add_subplot(1,n_dims,d+1, frameon=False)
		plot_func(ax, dim_to_show = d)
		ax.set_title('dim ' + str(d))

	fig.suptitle(title, fontsize=16)
	fig.savefig(plot_file_name + ".pdf")


def get_meshgrid(npts, int_y1, int_y2):
	min_y1, max_y1 = int_y1
	min_y2, max_y2 = int_y2
	
	y1_grid = np.linspace(min_y1, max_y1, npts)
	y2_grid = np.linspace(min_y2, max_y2, npts)

	xx, yy = np.meshgrid(y1_grid, y2_grid)

	flat_inputs = np.concatenate((np.expand_dims(xx.flatten(),1), np.expand_dims(yy.flatten(),1)), 1)
	flat_inputs = torch.from_numpy(flat_inputs).float()

	return xx, yy, flat_inputs


def add_white(cmap):
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# force the first color entry to be grey
	cmaplist[0] = (1.,1.,1.,1.0)
	# create the new map
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	return cmap



def plot_reconstructions(model, data_dict, experimentID = 0., itr = 0., 
	time_steps_to_predict = None, width = 5, mark_train_test = False):

	init_fonts(BIGGER_SIZE)

	data =  data_dict["data_to_predict"]
	mask = data_dict["mask_predicted_data"]
	if time_steps_to_predict is None:
		time_steps = data_dict["tp_to_predict"]
	else:
		time_steps = time_steps_to_predict
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	extrapolation = False
	intersect = [x for x in time_steps if x in observed_time_steps]
	if len(intersect) == 0:
		extrapolation = True

	n_dims = data.size(-1)

	max_y = data.cpu().numpy().max()
	min_y = data.cpu().numpy().min()

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	n_traj = min(data.size(0), 10)

	for traj_id in range(n_traj):
		height = 3 
		fig, ax_list = plt.subplots(1, n_dims, figsize=(width * n_dims, height), facecolor='white')

		one_traj = data[traj_id, :, :].unsqueeze(0)
		one_observed_traj = observed_data[traj_id, :, :].unsqueeze(0)
		one_observed_mask = observed_mask[traj_id, :, :].unsqueeze(0)

		n_traj_samples = 10
		reconstructions, info = model.get_reconstruction(
			time_steps, one_observed_traj, observed_time_steps,
			mask = one_observed_mask,
			n_traj_samples = n_traj_samples)
		reconstructions = reconstructions.squeeze(1)

		if reconstructions.size(0) == 1:
			n_traj_samples = 1


		for d in range(n_dims):
			# skip the reconstructions with only one point
			if (mask is not None) and (sum(mask[traj_id, :, d]) < 2):
				continue

			one_traj_d = one_traj[:,:,d]
			reconstructions_d = reconstructions[:,:,d]
			time_steps_d = time_steps

			observed_time_steps_d = observed_time_steps
			one_observed_traj_d =  one_observed_traj[:,:,d]

			if mask is not None:
				one_traj_d = torch.masked_select(one_traj_d[0,:], mask[traj_id, :, d].byte())
				time_steps_d = torch.masked_select(time_steps_d, mask[traj_id, :, d].byte())

				reconstructions_d = torch.masked_select(reconstructions_d, mask[traj_id, :, d].byte())				
				reconstructions_d = reconstructions_d.reshape(n_traj_samples, len(one_traj_d))
				
				one_traj_d = one_traj_d.unsqueeze(0)

			if observed_mask is not None:
				one_observed_traj_d = torch.masked_select(one_observed_traj_d[0,:], observed_mask[traj_id, :, d].byte())
				observed_time_steps_d = torch.masked_select(observed_time_steps, observed_mask[traj_id, :, d].byte())

				one_observed_traj_d = one_observed_traj_d.unsqueeze(0)

			one_traj_d = one_traj_d.unsqueeze(2)
			one_observed_traj_d = one_observed_traj_d.unsqueeze(2)
			reconstructions_d = reconstructions_d.unsqueeze(2)

			#mean_sq_error = np.round(torch.mean((one_traj_d - reconstructions_d)**2).cpu().numpy(),4)

			ax = ax_list if (n_dims == 1) else ax_list[d]
			
			if extrapolation:
				# Show also the data that we conditioned on
				enc_tp = torch.cat((observed_time_steps_d, data_dict["tp_to_predict"]))
				enc_data = torch.cat((one_observed_traj_d, one_traj_d), 1)
				plot_trajectories(ax, enc_data, enc_tp, marker='o', linestyle='')

				ax.axvline(x=time_steps_d[0].cpu().numpy())
			else:
				plot_trajectories(ax, one_traj_d, time_steps_d, marker='o', linestyle='')
			

			if extrapolation and ("reconstr_observed_data" in info):
				# For classic RNN, show recontructions also on the data that we conditioned on
				time_steps_d = torch.cat((observed_time_steps_d, time_steps_d))
			
				enc_data = info["reconstr_observed_data"][0, 0, :, 0]

				if observed_mask is not None:
					enc_data = torch.masked_select(enc_data, observed_mask[traj_id, :, d].byte())
				enc_data = enc_data.reshape(1,-1,1)

				reconstructions_d = torch.cat((enc_data, reconstructions_d), 1)

			plot_trajectories(ax, reconstructions_d, time_steps_d,  marker='', add_to_plot=True)
				# title="Reconstr sub data " + str(len(observed_time_steps)) + "/" + str(len(time_steps)) + \
				# 	" tp. Mean sq: " + str(mean_sq_error))

			bottom, top = ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5
			if mark_train_test:
				ax.fill_between((observed_time_steps[0].cpu().numpy(), observed_time_steps[-1].cpu().numpy()), 
					bottom, top, alpha=0.1, color = our_red)
				ax.fill_between((data_dict["tp_to_predict"][0].cpu().numpy(), data_dict["tp_to_predict"][-1].cpu().numpy()), 
					bottom, top, alpha=0.1, color = our_blue)

		fig.tight_layout()
		fig.savefig(dirname + "reconstr_{}_traj_{}_{}".format(experimentID, traj_id, itr) + ".pdf")
		plt.close(fig)
		plt.close()



def plot_reconstructions_per_patient(model, 
	data_dict, attr_list,
	experimentID = 0., itr = 0., n_traj_to_show = 10,
	attr_as_one_plot = True):

	SMALL_SIZE = 14
	MEDIUM_SIZE = 16
	BIGGER_SIZE = 18

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	extrapolation = False
	intersect = [x for x in time_steps if x in observed_time_steps]
	if len(intersect) == 0:
		extrapolation = True

	if isinstance(model, Classic_RNN) or isinstance(model, ODE_GRU_rnn):
		time_steps_to_predict = time_steps
	else:
		time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 50)
		time_steps_to_predict = time_steps_to_predict.to(get_device(data))

	n_traj_to_show = min(n_traj_to_show, data.size(0))
	for traj_id in range(n_traj_to_show):
		one_traj = data[traj_id, :, :]
		one_observed_traj = observed_data[traj_id, :, :]
		one_observed_mask = observed_mask[traj_id, :, :]
		one_predicted_mask = mask[traj_id, :, :]

		n_traj_samples = 10
		reconstructions, info = model.get_reconstruction(
			time_steps_to_predict, one_observed_traj.unsqueeze(0), observed_time_steps,
			mask = one_observed_mask.unsqueeze(0),
			n_traj_samples = n_traj_samples)
		reconstructions = reconstructions.squeeze(1)

		non_zero_attributes = (torch.sum(one_predicted_mask,0) >= 1).cpu().numpy()

		non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
		n_non_zero = sum(non_zero_attributes)

		one_predicted_mask = one_predicted_mask[:, non_zero_idx]
		one_traj = one_traj[:, non_zero_idx]
		reconstructions = reconstructions[:,:, non_zero_idx]

		one_observed_traj = one_observed_traj[:, non_zero_idx]
		one_observed_mask = one_observed_mask[:, non_zero_idx]

		if "log_lambda_y" in info:
			# lambdas for one trajectory
			lambdas = info["log_lambda_y"]
			lambdas = torch.exp(lambdas)
			lambdas = lambdas[:,:,:,non_zero_idx]

		params_non_zero = [attr_list[i] for i in non_zero_idx]
		params_dict = {k: i for i, k in enumerate(params_non_zero)}

		if attr_as_one_plot:
			n_col = 3
			n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
			width = 15
			height = 4 * n_row
			fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

		for i in range(n_non_zero):
			param = params_non_zero[i]
			param_id = params_dict[param]

			if torch.sum(one_predicted_mask[:,param_id]) == 0.:
				next

			tp_mask = one_predicted_mask[:,param_id].long()
			tp_cur_param = time_steps[tp_mask == 1.]

			data_cur_param = one_traj[tp_mask == 1., param_id]
			reconstr_cur_param = reconstructions[:,:, param_id]

			data_cur_param = data_cur_param.reshape(1,-1,1)
			reconstr_cur_param = reconstr_cur_param.unsqueeze(-1)

			tp_mask = one_observed_mask[:,param_id].long()
			observed_tp_cur_param = observed_time_steps[tp_mask == 1.]
			enc_data_cur_param = one_observed_traj[tp_mask == 1., param_id].reshape(1,-1,1)
			enc_data_cur_param = enc_data_cur_param.reshape(1,-1,1)

			if attr_as_one_plot:
				if n_row == 1:
					ax = ax_list[i % n_col]
				else:
					ax = ax_list[i // n_col, i % n_col]
				ax_rec = ax_pois = ax
			else:
				width = 5
				height = 3
				# Two plots per attribute: one for reconstr, the other one for Poisson
				gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

				ax_rec = plt.subplot(gs[0])
				ax_pois = plt.subplot(gs[1])


			if extrapolation:
				enc_tp = torch.cat((observed_tp_cur_param, tp_cur_param))
				enc_data = torch.cat((enc_data_cur_param, data_cur_param), 1)

				plot_trajectories(ax_rec, enc_data, enc_tp, marker='o', linestyle='',
					min_y = -0.1, max_y = 1.1)
				ax_rec.axvline(x=tp_cur_param[0].cpu().numpy())
			else:
				plot_trajectories(ax_rec, data_cur_param, tp_cur_param, marker='o', linestyle='',
					min_y = -0.1, max_y = 1.1)

			if "log_lambda_y" in info:
				lambdas_cur = lambdas[:, 0, :, param_id].unsqueeze(-1)
				
				if attr_as_one_plot:
					lambdas_cur = lambdas_cur / torch.max(lambdas_cur) * 0.3
				
				plot_trajectories(ax_pois, lambdas_cur, time_steps_to_predict, 
					marker='', linestyle = '-.', add_to_plot=True)

			plot_trajectories(ax_rec, reconstr_cur_param, time_steps_to_predict, 
				min_y = -0.1, max_y = 1.1,
				marker='', add_to_plot=True)
			ax_rec.set_title(param)

			if "log_lambda_y" in info:
				poisson_integral = torch.mean(info["int_lambda"][:, 0, param_id]).cpu().numpy()
				if attr_as_one_plot:
					ax_pois.set_title("{} : {} tp. Poisson int: {:.3f}".format(
						param, len(tp_cur_param), np.round(poisson_integral,3)))
				else:
					ax_pois.set_title("Poisson rate")
					ax_pois.set_ylabel('lambda')

			if not attr_as_one_plot:
				plt.tight_layout()
				plt.savefig(dirname + "reconstr_traj_{}_{}_{}_{}".format(traj_id, param, experimentID, itr) + ".pdf")
				plt.close()


		if attr_as_one_plot:
			fig.tight_layout()
			fig.savefig(dirname + "reconstr_traj_{}_{}_{}".format(traj_id, experimentID, itr) + ".pdf")
			plt.close(fig)
	plt.close()







def plot_non_missing_parameters(data, mask, time_steps, attr_list, plot_name, 
	reconstructions = None, reconstr_time_steps = None, vlines = None):
	
	plt.close()

	SMALL_SIZE = 14
	MEDIUM_SIZE = 16
	BIGGER_SIZE = 18

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

	non_zero_attributes = (torch.sum(mask,0) >= 1).cpu().numpy()
	non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
	n_non_zero = sum(non_zero_attributes)

	mask = mask[:, non_zero_idx]
	data = data[:, non_zero_idx]

	params_non_zero = [attr_list[i] for i in non_zero_idx]
	params_dict = {k: i for i, k in enumerate(params_non_zero)}

	n_row = n_non_zero
	width = 15
	height = 0.7 * n_row
	fig, ax_list = plt.subplots(n_row, 1, figsize=(width, height), 
		facecolor='white', sharex='col', frameon=False)

	# plt.axis('on')
	# fig = plt.figure(figsize = (width, height))
	# gs1 = gridspec.GridSpec(n_row, 1)
	# gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

	for i in range(n_non_zero):
		print("parameter " + str(i))

		param = params_non_zero[i]
		param_id = params_dict[param]

		if torch.sum(mask[:,param_id]) == 0.:
			next

		tp_mask = mask[:,param_id].long()
		tp_cur_param = time_steps[tp_mask == 1.]

		data_cur_param = data[tp_mask == 1., param_id]
		data_cur_param = data_cur_param.reshape(1,-1,1)

		ax_rec = ax = ax_list[i]

		plot_trajectories(ax_rec, data_cur_param, tp_cur_param, 
				marker='o', linestyle='',
				min_y = -0.1, max_y = 1.1)

		if reconstructions is not None: 
			reconstructions = reconstructions[:,:, non_zero_idx]
			reconstr_cur_param = reconstructions[:,:, param_id]
			reconstr_cur_param = reconstr_cur_param.unsqueeze(-1)
			plot_trajectories(ax_rec, reconstr_cur_param, reconstr_time_steps, 
				min_y = -0.1, max_y = 1.1,
				marker='', add_to_plot=True)

		if vlines is not None:
			for x in vlines:
				ax_rec.axvline(x=x, linestyle = "dashed", alpha = 0.3)

		#ax_rec.axis('on')
		plt.rc('axes', titlesize=5)     # fontsize of the axes title
		plt.rc('axes', labelsize=5)    # fontsize of the x and y labels

		ax_rec.set_yticklabels([])
		ax_rec.set_xlabel("Time")
		ax_rec.set_ylabel(param, rotation = 0,  labelpad=40)
		ax_rec.set_title(None)

	fig.subplots_adjust(wspace=0, hspace=0)
	#fig.tight_layout()
	fig.savefig(plot_name)
	plt.close(fig)
	plt.close()









def plot_reconstructions_per_patient_v2(model, 
	data_dict, attr_list,
	experimentID = 0., itr = 0., n_traj_to_show = 10,
	quantization = 1.):

	time_scale = 48.
	n_bins = int(time_scale / quantization)

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	extrapolation = False
	intersect = [x for x in time_steps if x in observed_time_steps]
	if len(intersect) == 0:
		extrapolation = True

	if isinstance(model, Classic_RNN) or isinstance(model, ODE_GRU_rnn):
		time_steps_to_predict = time_steps
	else:
		time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 50)
		time_steps_to_predict = time_steps_to_predict.to(get_device(data))

	n_traj_to_show = min(n_traj_to_show, data.size(0))
	for traj_id in range(n_traj_to_show):

		print("traj_id " + str(traj_id))


		one_traj = data[traj_id, :, :]
		one_observed_traj = observed_data[traj_id, :, :]
		one_observed_mask = observed_mask[traj_id, :, :]
		one_predicted_mask = mask[traj_id, :, :]

		n_traj_samples = 10
		reconstructions, info = model.get_reconstruction(
			time_steps_to_predict, one_observed_traj.unsqueeze(0), observed_time_steps,
			mask = one_observed_mask.unsqueeze(0),
			n_traj_samples = n_traj_samples)
		reconstructions = reconstructions.squeeze(1)

		# if "log_lambda_y" in info:
		# 	# lambdas for one trajectory
		# 	lambdas = info["log_lambda_y"]
		# 	lambdas = torch.exp(lambdas)
		# 	lambdas = lambdas[:,:,:,non_zero_idx]

		if extrapolation:
			data_for_plotting = torch.cat((one_observed_traj, one_traj), -1)
			mask_for_plotting = torch.cat((one_observed_mask, one_predicted_mask), -1)
			tp_for_plotting = torch.cat((observed_time_steps, time_steps), 0)
		else:
			data_for_plotting = one_traj
			mask_for_plotting = one_predicted_mask
			tp_for_plotting = time_steps
		
		tp_for_plotting = tp_for_plotting * time_scale

		plot_name = dirname + "reconstr_traj_{}_{}_{}".format(traj_id, experimentID, itr) + ".pdf"
		
		plot_non_missing_parameters(
			data_for_plotting, mask_for_plotting, tp_for_plotting, 
			attr_list, plot_name)
			# reconstructions = reconstructions,
			# reconstr_time_steps = time_steps_to_predict * time_scale)
			#vlines = np.linspace(0, time_scale, n_bins))

		# if "log_lambda_y" in info:
		# 	lambdas_cur = lambdas[:, 0, :, param_id].unsqueeze(-1)
			
		# 	lambdas_cur = lambdas_cur / torch.max(lambdas_cur) * 0.3
			
		# 	plot_trajectories(ax_pois, lambdas_cur, time_steps_to_predict, 
		# 		marker='', linestyle = '-.', add_to_plot=True)
		
		# if "log_lambda_y" in info:
		# 	poisson_integral = torch.mean(info["int_lambda"][:, 0, param_id]).cpu().numpy()
		# 	ax_pois.set_title("{} : {} tp. Poisson int: {:.3f}".format(
		# 		param, len(tp_cur_param), np.round(poisson_integral,3)))
			







def plot_data_missingness_physionet(model, 
	data_dict, attr_list,
	experimentID = 0., itr = 0., n_traj_to_show = 10,
	quantization = 1.):

	TINY_SIZE = 6

	plt.rc('font', size=TINY_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=TINY_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=TINY_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=TINY_SIZE)  # fontsize of the figure title

	time_scale = 48.
	n_bins = int(time_scale / quantization)

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	extrapolation = False
	intersect = [x for x in time_steps if x in observed_time_steps]
	if len(intersect) == 0:
		extrapolation = True

	n_traj_to_show = min(n_traj_to_show, data.size(0))
	for traj_id in range(n_traj_to_show):

		print("traj_id " + str(traj_id))

		one_traj = data[traj_id, :, :]
		one_observed_traj = observed_data[traj_id, :, :]
		one_observed_mask = observed_mask[traj_id, :, :]
		one_predicted_mask = mask[traj_id, :, :]

		# if "log_lambda_y" in info:
		# 	# lambdas for one trajectory
		# 	lambdas = info["log_lambda_y"]
		# 	lambdas = torch.exp(lambdas)
		# 	lambdas = lambdas[:,:,:,non_zero_idx]

		if extrapolation:
			data_for_plotting = torch.cat((one_observed_traj, one_traj), -1)
			mask_for_plotting = torch.cat((one_observed_mask, one_predicted_mask), -1)
			tp_for_plotting = torch.cat((observed_time_steps, time_steps), 0)
		else:
			data_for_plotting = one_traj
			mask_for_plotting = one_predicted_mask
			tp_for_plotting = time_steps

		tp_for_plotting = tp_for_plotting * time_scale

		#plt.figure(figsize = (5,3))
		plt.figure(figsize = (5,1.3))

		n_tp, n_dims = mask_for_plotting.size()
		df = pd.DataFrame(mask_for_plotting.cpu().transpose(0,1).numpy())

		# subset the table
		n_vars_to_show = 7
		ind = [i for i in range(len(attr_list)) if attr_list[i] == "K"][0] #Platelets

		yticklabels = attr_list[ind:(ind+n_vars_to_show)]
		df = df.iloc[ind:(ind+n_vars_to_show)]

		df.columns = np.round(np.linspace(0, time_scale, n_tp),1)

		ax = sns.heatmap(df, cmap=["white", "black"], 
			cbar = False, yticklabels = yticklabels, 
			xticklabels = int(1/quantization)*5)#yticklabels = False)
		ax.set_xlabel("Time")
		ax.xaxis.labelpad = 0.3
		ax.yaxis.labelpad = 0.3
		ax.tick_params(axis='both', which='major', pad=-0.2)

		# Make a frame around the heatmap
		for _, spine in ax.spines.items():
			spine.set_visible(True)
			spine.set_linewidth(0.1)

		plot_name = dirname + "missingness_traj_{}_{}_{}".format(traj_id, experimentID, itr) + ".pdf"
		plt.tight_layout()
		plt.savefig(plot_name)
		plt.close()


		# if "log_lambda_y" in info:
		# 	lambdas_cur = lambdas[:, 0, :, param_id].unsqueeze(-1)
			
		# 	lambdas_cur = lambdas_cur / torch.max(lambdas_cur) * 0.3
			
		# 	plot_trajectories(ax_pois, lambdas_cur, time_steps_to_predict, 
		# 		marker='', linestyle = '-.', add_to_plot=True)
		
		# if "log_lambda_y" in info:
		# 	poisson_integral = torch.mean(info["int_lambda"][:, 0, param_id]).cpu().numpy()
		# 	ax_pois.set_title("{} : {} tp. Poisson int: {:.3f}".format(
		# 		param, len(tp_cur_param), np.round(poisson_integral,3)))
			















def plot_ode_performance(model, data_dict,
	experimentID = 0., itr = 0., n_traj_to_show = 20):

	init_fonts(MEDIUM_SIZE)

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]


	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 100)
	time_steps_to_predict = time_steps_to_predict.to(get_device(data))
	t_max = time_steps_to_predict[-1].cpu().numpy()
	t_min = time_steps_to_predict[0].cpu().numpy()

	n_traj_to_show = min(n_traj_to_show, data.size(0))

	# width = 5 * n_traj_to_show
	# height = 3 

	# fig, ax_list = plt.subplots(1, n_traj_to_show, figsize=(width, height), facecolor='white')

	for traj_id in range(n_traj_to_show):
		one_traj = data[traj_id, :, :].unsqueeze(0)
		one_observed_traj = observed_data[traj_id, :, :].unsqueeze(0)
		one_observed_mask = observed_mask[traj_id, :, :].unsqueeze(0)

		n_traj_samples = 10
		reconstructions, info = model.get_reconstruction(
			time_steps_to_predict, one_observed_traj, observed_time_steps,
			mask = one_observed_mask,
			n_traj_samples = n_traj_samples)
		reconstructions = reconstructions.squeeze(1)

		#n_plots = 2
		#plt.box(on=None)
		# fig, ax_list = plt.subplots(n_plots, 1, figsize=(7, 1.5 * n_plots), facecolor='white')

		# ax_norms, ax_hist = ax_list
		# ax_norms.axis('on')
		# ax_hist.axis('on')

		fig = plt.figure(figsize=(7, 1.5))
		ax_norms = fig.add_subplot(111, frameon=False)
		
		#norms = info["ode_func_norms"] 
		z = info["latent_traj"]

		# Collect the ODE gradients for latents z
		f_z = []
		for i in range(len(time_steps_to_predict)):
			f_z.append(model.diffeq_solver.ode_func(time_steps_to_predict[i], z[:,:,i,:]))
		f_z = torch.stack(f_z).permute(1,2,0,3)

		# Take mean over all trajectory samples
		norms = torch.mean(torch.norm(f_z, dim=-1),0).squeeze(0)
		ax_norms.plot(time_steps_to_predict.cpu().numpy()/t_max, norms.cpu().numpy())

		ax_norms.set_xlim(0, 1)
		ax_norms.axis('on')

		fig.tight_layout()
		fig.subplots_adjust(hspace=0.1, wspace=0.05)
		fig.savefig(dirname + "ode_norm_traj_{}_{}_{}".format(traj_id, experimentID, itr) + ".pdf")
		plt.close(fig)

		#ax.set_title("Traj {} # ode func evals: {}".format(traj_id, len(info["ode_func_ts"])))
		#ax_norms.set_xlabel('time')
		#ax_norms.set_ylabel('||f(z)||')
		#ax_norms.set_ylabel('||f(z)||')

		fig = plt.figure(figsize=(7, 1.7))
		ax_hist = fig.add_subplot(111, frameon=False)

		tp_for_hist_plot = torch.stack((time_steps_to_predict[0], time_steps_to_predict[-1]))

		reconstructions, info = model.get_reconstruction(
			tp_for_hist_plot, one_observed_traj, observed_time_steps,
			mask = one_observed_mask,
			n_traj_samples = n_traj_samples)

		# Show the histogram of observation times
		#kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
		sns.distplot(info["ode_func_ts"].cpu().numpy()/t_max, color="dodgerblue", 
			ax = ax_hist, kde=False, bins = 25) # axlabel='time', 
		sns.despine(left=True, bottom=True, right=True)
		#ax_hist.set_xlabel('time')
		#ax_hist.set_ylabel('# evals')
		ax_hist.set_xlim(0, 1)
		ax_hist.axis('on')

		fig.tight_layout()
		fig.subplots_adjust(hspace=0.1, wspace=0.05)
		fig.savefig(dirname + "ode_func_evals_hist_traj_{}_{}_{}".format(traj_id, experimentID, itr) + ".pdf")
		plt.close(fig)



def plot_y0_space(model, data_dict, dataset_name,
	experimentID = 0., itr = 0., n_traj_to_show = 100,
	color_by_first_tp = False, use_umap = False):

	init_fonts(BIGGER_SIZE)

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	# n_traj_to_show = min(n_traj_to_show, data.size(0))
	# fig, ax_list = plt.subplots(1, n_traj_to_show, figsize=(5, 5), facecolor='white')

	reconstructions, info = model.get_reconstruction(
		time_steps, observed_data, observed_time_steps,
		mask = observed_mask,
		n_traj_samples = 1)
	reconstructions = reconstructions.squeeze(1)

	first_point_mu, first_point_std, first_point_enc = info["first_point"]
	n_latent_dims = first_point_enc.size(-1)

	os.makedirs(dirname + "y0_space", exist_ok=True)

	plt.close()

	# Plot PCA
	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, frameon=False)
	plt.figure(figsize=(5,4))
	ax = plt.gca()
	ax.cla()
	
	if use_umap:
		y0_reduced = umap.UMAP().fit_transform(first_point_enc.cpu().numpy()[0])
	else:
		y0_reduced = PCA(n_components=2).fit_transform(first_point_enc.cpu().numpy()[0])
	

	if color_by_first_tp:
		# non_missing = np.argwhere(observed_mask[:,0,0].cpu().numpy() != 0.)
		# color_by = observed_data[:,0,0][non_missing].cpu().numpy()

		#color by the first point -- if the value is not missing in at least first five time points
		non_missing = np.argwhere(torch.sum(observed_mask[:,:5,0],-1).cpu().numpy() != 0.)
		color_by = torch.max(torch.abs(observed_data[:,:5,0]),-1)[0][non_missing].cpu().numpy()

		ax.scatter(
			y0_reduced[:, 0][non_missing], 
			y0_reduced[:, 1][non_missing], 
			c = color_by, 
			s = 30, alpha = 0.8)
	else:
		# first_point_enc shape: [1, n_traj, n_dims]
		ax.scatter(y0_reduced[:, 0], y0_reduced[:, 1], s = 30)

	#ax.set_title("Latent space y0 (PCA)")
	ax.set_xlabel('')
	ax.set_ylabel('')
	if not use_umap:
		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')
	ax.legend()
	#fig.legend(loc=7) 

	plt.tight_layout()
	plt.savefig(dirname + "y0_space/y0_space_{}_{}_{}".format(
		"UMAP" if use_umap else "PCA", experimentID, itr) + ".pdf")
	plt.close()







def plot_y0_space_hopper(model, data_dict,
	experimentID = 0., itr = 0., n_traj_to_show = 100):

	init_fonts()

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]


	print("observed_data")
	print(observed_data.size())


	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	# n_traj_to_show = min(n_traj_to_show, data.size(0))
	# fig, ax_list = plt.subplots(1, n_traj_to_show, figsize=(5, 5), facecolor='white')

	reconstructions, info = model.get_reconstruction(
		time_steps, observed_data, observed_time_steps,
		mask = observed_mask,
		n_traj_samples = 1)
	reconstructions = reconstructions.squeeze(1)

	first_point_mu, first_point_std, first_point_enc = info["first_point"]
	n_latent_dims = first_point_enc.size(-1)

	os.makedirs(dirname + "y0_space", exist_ok=True)

	plt.close()

	# Plot PCA
	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, frameon=False)
	plt.figure(figsize=(5,4))
	ax = plt.gca()
	ax.cla()
	

	#y0_umap = PCA(n_components=2).fit_transform(first_point_enc.cpu().numpy()[0])
	y0_umap = umap.UMAP().fit_transform(first_point_enc.cpu().numpy()[0])


	for i in range(14):
		plt.close()
		plt.figure(figsize=(5,4))
		ax = plt.gca()
		ax.cla()
		
		# fig = plt.figure(figsize=(4, 4), facecolor='0.75')
		# ax = fig.add_subplot(111, frameon=False)
		
		non_missing = np.argwhere(observed_mask[:,0,i].cpu().numpy() != 0.)

		ax.scatter(
			y0_umap[:, 0][non_missing], 
			y0_umap[:, 1][non_missing], 
			c = observed_data[:,0,i][non_missing].cpu().numpy(),
			s = 40, alpha = 0.8, cmap = 'plasma')#cmap = 'rainbow')

		#ax.set_title("Latent space y0 (PCA)")
		ax.set_xlabel('h1')
		ax.set_ylabel('h2')

		plt.tight_layout()
		plt.savefig(dirname + "y0_space/y0_space_pos{}_UMAP_{}_{}".format(
			i, experimentID, itr) + ".pdf")
		plt.close()

	# fig = plt.figure(figsize=(4, 4), facecolor='gray')
	# ax = fig.add_subplot(111, frameon=False)
	plt.figure(figsize=(4,4))
	ax = plt.gca()
	ax.cla()
	ax.scatter(y0_umap[:, 0], y0_umap[:, 1], alpha=0.3)

	#ax.set_title("Latent space y0 (PCA)")
	ax.set_xlabel('h1')
	ax.set_ylabel('h2')
	ax.legend()

	plt.tight_layout()
	plt.savefig(dirname + "y0_space/y0_space_UMAP_{}_{}".format(
		experimentID, itr) + ".pdf")
	plt.close()





def plot_y0_space_activity(model, data_dict,
	experimentID = 0., itr = 0., n_traj_to_show = 100):

	SMALL_SIZE = 14
	MEDIUM_SIZE = 16
	BIGGER_SIZE = 18
	LARGE_SIZE = 22

	plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	# n_traj_to_show = min(n_traj_to_show, data.size(0))
	# fig, ax_list = plt.subplots(1, n_traj_to_show, figsize=(5, 5), facecolor='white')

	reconstructions, info = model.get_reconstruction(
		time_steps, observed_data, observed_time_steps,
		mask = observed_mask,
		n_traj_samples = 1)
	reconstructions = reconstructions.squeeze(1)

	latent_states = info["latent_traj"]
	latent_states = latent_states.reshape(-1, latent_states.size(-1))

	os.makedirs(dirname + "y0_space", exist_ok=True)
	plt.close()

	# Plot PCA
	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, frameon=False)
	
	fig = plt.figure(figsize=(10,8))
	ax = fig.add_subplot(111)
	ax.cla()

	#y0_umap = PCA(n_components=2).fit_transform(first_point_enc.cpu().numpy()[0])
	y0_umap = umap.UMAP().fit_transform(latent_states.cpu().numpy())

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		print(data_dict["labels"].size())

		y0_labels = torch.argmax(data_dict["labels"],-1)
		y0_labels = y0_labels.reshape(-1)
		y0_labels = y0_labels.cpu().numpy()

		for lab in range(len(PersonActivity.label_names)):
			y0 = y0_umap[y0_labels == lab,:]
			ax.scatter(y0[:, 0], y0[:, 1],
						label = str(PersonActivity.label_names[lab]),
						alpha=0.9, cmap = 'hsv')

	#ax.set_title("Latent space y0 (PCA)")
	ax.set_xlabel('h1')
	ax.set_ylabel('h2')
	#ax.legend()
	fig.legend(loc=7)
	fig.tight_layout()
	fig.subplots_adjust(right=0.75)

	plt.savefig(dirname + "y0_space/y0_space_UMAP_{}_{}".format(
		experimentID, itr) + ".pdf")
	plt.close()




def plot_y0_space_physionet(model, data_dict,
	experimentID = 0., itr = 0., n_traj_to_show = 100):

	init_fonts()

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]


	print("observed_data")
	print(observed_data.size())


	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	# n_traj_to_show = min(n_traj_to_show, data.size(0))
	# fig, ax_list = plt.subplots(1, n_traj_to_show, figsize=(5, 5), facecolor='white')

	reconstructions, info = model.get_reconstruction(
		time_steps, observed_data, observed_time_steps,
		mask = observed_mask,
		n_traj_samples = 1)
	reconstructions = reconstructions.squeeze(1)

	first_point_mu, first_point_std, first_point_enc = info["first_point"]
	n_latent_dims = first_point_enc.size(-1)

	os.makedirs(dirname + "y0_space", exist_ok=True)

	plt.close()

	# Plot PCA
	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, frameon=False)
	plt.figure(figsize=(5,4))
	ax = plt.gca()
	ax.cla()

	#y0_umap = PCA(n_components=2).fit_transform(first_point_enc.cpu().numpy()[0])
	y0_umap = umap.UMAP().fit_transform(first_point_enc.cpu().numpy()[0])

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		colors = [our_blue, our_red]
		labels = data_dict["labels"].cpu().numpy()

		for mort in [0, 1]:
			y0 = y0_umap[labels == mort,:]
			# first_point_enc shape: [1, n_traj, n_dims]
			ax.scatter(y0[:, 0], y0[:, 1],
						c = colors[mort],
						label = "Mortality = {}".format(mort),
						alpha=0.3)

	#ax.set_title("Latent space y0 (PCA)")
	ax.set_xlabel('h1')
	ax.set_ylabel('h2')
	ax.legend()

	plt.tight_layout()
	plt.savefig(dirname + "y0_space/y0_space_UMAP_{}_{}".format(
		experimentID, itr) + ".pdf")
	plt.close()





def plot_traj_from_prior(model, time_steps,
	experimentID = 0., itr = 0., n_traj_to_show = 100, ylim = None):

	init_fonts(MEDIUM_SIZE)

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	plt.close()

	fig = plt.figure(figsize=(5, 3))
	ax = fig.add_subplot(111, frameon=False)
	ax.cla()

	traj_from_prior = model.sample_traj_from_prior(time_steps, n_traj_samples = 6)
	# Since in this case n_traj = 1, n_traj_samples -- requested number of samples from the prior, squeeze n_traj dimension
	traj_from_prior = traj_from_prior.squeeze(1)

	plot_trajectories(ax, traj_from_prior, time_steps, marker = '')
	ax.set_ylim(ylim)

	plt.tight_layout()
	plt.savefig(dirname + "traj_from_prior_{}_{}".format(
		experimentID, itr) + ".png")
	plt.close()



# def plot_reconstruct_diff_t0(model, data, time_steps, 
# 	experimentID = 0., itr = 0., n_tp_to_sample = None):
# 	# sample at the original time points

# 	n_plots = 10
# 	width = 4 * n_plots
# 	height = 3

# 	max_y = data.cpu().numpy().max()
# 	min_y = data.cpu().numpy().min()

# 	fig, ax_list = plt.subplots(1, n_plots, sharex='col', sharey='row', figsize=(width, height), facecolor='white')

# 	int_length = time_steps[-1] - time_steps[0]
# 	t0_options = utils.linspace_vector(time_steps[0] - int_length, time_steps[0]-0.01, n_plots) 

# 	for traj_id in range(5):
# 		one_traj = data[traj_id].unsqueeze(0)

# 		observed_data, observed_time_steps = utils.subsample_timepoints(one_traj, time_steps, n_tp_to_sample = n_tp_to_sample)

# 		for i, t0 in enumerate(t0_options):
# 			time_steps_to_predict = torch.cat((t0.reshape(-1), time_steps), 0)

# 			reconstructions, info = model.get_reconstruction(
# 				time_steps_to_predict, observed_data, observed_time_steps, 
# 				n_traj_samples = 10, t0=t0)
# 			reconstructions = reconstructions.squeeze(1)
# 			mean_sq_error = np.round(torch.mean((one_traj - reconstructions[:,1:,:])**2).cpu().numpy(),4)

# 			plot_trajectories(ax_list[i], one_traj, time_steps, marker='o', linestyle='')
# 			plot_trajectories(ax_list[i], reconstructions, time_steps_to_predict, 
# 				title="t0: " + str(t0.cpu().numpy()) + ". Mean sq: " + str(mean_sq_error),  marker='o', add_to_plot=True)

# 		dirname = "plots/" + str(experimentID) + "/"
# 		os.makedirs(dirname, exist_ok=True)

# 		fig.tight_layout()
# 		fig.savefig(dirname + "reconstr_diff_t0_traj_{}_{}_{:03d}".format(traj_id, experimentID, itr) + ".pdf")
# 		plt.close(fig)



def plot_reconstruct_encoding_t0_ti(model, data, time_steps, 
	experimentID = 0., itr = 0.):
	# Get the encoding of y(t0) using only observations t0 ... ti
	# Reconstruct the data from y(t0)
	# Look at the variance over y(t0) and over the decoded trajectory

	init_fonts(MEDIUM_SIZE)

	n_plots = 3
	tp_range = torch.linspace(10, len(time_steps), n_plots).type(torch.IntTensor)

	width = 4 * n_plots
	height = 3

	fig, ax_list = plt.subplots(1, n_plots, sharex='col', sharey='row', figsize=(width, height), facecolor='white')

	for traj_id in range(10):
		one_traj = data[traj_id].unsqueeze(0)

		for i, ti_index in enumerate(tp_range):
			observed_data = one_traj[:,:ti_index]
			observed_time_steps = time_steps[:ti_index]

			reconstructions, info = model.get_reconstruction(
				time_steps, observed_data, observed_time_steps,
				mask = torch.ones_like(observed_data).to(get_device(data)), 
				n_traj_samples = 10)
			reconstructions = reconstructions.squeeze(1)

			mean_sq_error = np.round(torch.mean((one_traj - reconstructions)**2).cpu().numpy(),4)

			plot_trajectories(ax_list[i], observed_data, observed_time_steps, marker='o', linestyle='')
			plot_trajectories(ax_list[i], reconstructions, time_steps, 
				title= str(ti_index.cpu().numpy()) + " observed points", # Mean sq: " + str(mean_sq_error),  
				marker='', add_to_plot=True) # marker='o'

		dirname = "plots/" + str(experimentID) + "/"
		os.makedirs(dirname, exist_ok=True)

		fig.tight_layout()
		fig.savefig(dirname + "reconstr_encoding_t0_ti_traj_{}_{}_{}".format(traj_id, experimentID, itr) + ".pdf")
		plt.close(fig)
		plt.close()




def plot_h0_stdev(model, data, time_steps, 
	experimentID = 0., itr = 0.):
	# Get the encoding of y(t0) using only observations t0 ... ti
	# Reconstruct the data from y(t0)
	# Look at the variance over y(t0) and over the decoded trajectory

	init_fonts(BIGGER_SIZE)

	plt.rc('text', usetex=True)

	data = data[:1000]
	print("data_size")
	print(data.size())

	n_measurements = 10
	n_tp_to_subsample = torch.linspace(10, len(time_steps), n_measurements).type(torch.IntTensor)

	#fig = plt.figure(figsize=(4.5, 2.5))
	fig = plt.figure(figsize=(3.5, 3))
	ax = fig.add_subplot(111, frameon=False)

	metrics = []
	for i, max_tp in enumerate(n_tp_to_subsample):
		observed_data = data[:,:max_tp]
		observed_time_steps = time_steps[:max_tp]

		reconstructions, info = model.get_reconstruction(
			time_steps, observed_data, observed_time_steps,
			mask = torch.ones_like(observed_data).to(get_device(data)), 
			n_traj_samples = 10)
		reconstructions = reconstructions.squeeze(1)

		first_point_mu, first_point_std, first_point_enc = info["first_point"]

		norm = torch.norm(first_point_std[0], dim=-1)

		distr = Independent(Normal(loc = first_point_mu.squeeze(0), scale = first_point_std.squeeze(0)), 1)
		entropy = distr.entropy()
		
		metrics.append(norm)

	metrics = torch.stack(metrics, 0)

	# mean over different trajectories
	metrics_mean = torch.mean(metrics, 1)

	low_percentile = np.percentile(metrics.cpu().numpy(), 10, axis=1)
	high_percentile = np.percentile(metrics.cpu().numpy(), 90, axis=1)


	ax.plot(n_tp_to_subsample.cpu().numpy(), metrics_mean.cpu().numpy(), marker='o')
	ax.fill_between(n_tp_to_subsample.cpu().numpy(), low_percentile, high_percentile, alpha=0.3)

	# for i in range(data.size(0)):
	# 	ax.plot(n_tp_to_subsample.cpu().numpy(), metrics[:,i].cpu().numpy(), marker='o')

	ax.set_xlabel("Number of time points")
	#ax.set_ylabel(r'$|| \Sigma_{z_0} ||$')
	ax.set_ylabel(r'$\mathrm{H}[z_0]$')
	ax.axis('on')

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	fig.tight_layout()
	fig.savefig(dirname + "h0_entropy_traj_{}_{}".format(experimentID, itr) + ".pdf")
	plt.close(fig)
	plt.close()
	plt.rc('text', usetex=False)







def plot_n_ode_calls_versus_n_points(model, data, time_steps, 
	experimentID = 0., itr = 0.):
	# Get the encoding of y(t0) using only observations t0 ... ti
	# Reconstruct the data from y(t0)
	# Look at the variance over y(t0) and over the decoded trajectory

	init_fonts()

	n_traj_to_show = 20
	n_measurements = 10

	width = 5
	height = 5

	# # Number of ODE function eval based on number of CONDITIONING time points
	# fig = plt.figure(figsize=(5, 5), facecolor='white')
	# ax = fig.add_subplot(111, frameon=False)

	# for traj_id in range(1):
	# 	one_traj = data[traj_id].unsqueeze(0)

	# 	d = {"n_calls": [], "n_points": []}
	# 	n_tp_to_choose = np.linspace(5, len(time_steps), n_measurements).astype(int)
	# 	for i in n_tp_to_choose:
	# 		observed_tp_indices = sorted(np.random.choice(len(time_steps), i, replace = False))

	# 		observed_data = one_traj[:,observed_tp_indices]
	# 		observed_time_steps = time_steps[observed_tp_indices]

	# 		reconstructions, info = model.get_reconstruction(
	# 			time_steps, observed_data, observed_time_steps,
	# 			mask = torch.ones_like(observed_data).to(get_device(data)), 
	# 			n_traj_samples = 10)
	# 		reconstructions = reconstructions.squeeze(1)

	# 		d["n_calls"].append(info["n_calls"]) 
	# 		d["n_points"].append(i) 

		
	# ax.scatter(d["n_points"], d["n_calls"], s = 10, marker='o')
	# ax.set_xlabel('# conditioned time points')
	# ax.set_ylabel('# ODE function calls')

	# dirname = "plots/" + str(experimentID) + "/"
	# os.makedirs(dirname, exist_ok=True)

	# fig.tight_layout()
	# fig.savefig(dirname + "ode_func_calls_n_observed_tp_{}_{}".format(experimentID, itr) + ".pdf")
	# plt.close(fig)

	##########################
	# Number of ODE function eval based LENGTH OF THE INTERVAL
	fig = plt.figure(figsize=(5, 5), facecolor='white')
	ax = fig.add_subplot(111, frameon=False)

	for traj_id in range(n_traj_to_show):
		one_traj = data[traj_id].unsqueeze(0)

		d = {"n_calls": [], "int_length": []}
		n_tp_to_choose = np.linspace(5, len(time_steps), n_measurements).astype(int)
		for i in n_tp_to_choose:
			#idx = sorted(np.random.choice(len(time_steps), i, replace = False))
			#time_steps_to_predict = time_steps[idx]
			time_steps_to_predict = time_steps[:i]
			if time_steps_to_predict[0] != 0:
				time_steps_to_predict = torch.cat((time_steps[:1], time_steps[idx]),0)

			observed_data = one_traj
			observed_time_steps = time_steps

			reconstructions, info = model.get_reconstruction(
				time_steps_to_predict, observed_data, observed_time_steps,
				mask = torch.ones_like(observed_data).to(get_device(data)), 
				n_traj_samples = 10)
			reconstructions = reconstructions.squeeze(1)

			d["n_calls"].append(info["n_calls"]) 
			d["int_length"].append(float(i) / len(time_steps) * 100) 
		
		ax.plot(d["int_length"], d["n_calls"], marker='o')
	
	ax.set_ylim(0, len(time_steps)+3)
	# plot the diagonal line
	low_x, high_x = ax.get_xlim()
	low_y, high_y = ax.get_ylim()
	low = max(low_x, low_y)
	high = min(high_x, high_y)
	ax.plot([low,high], [low,high], ls="--", c=".3")

	ax.set_xlabel("% interval length (" + str(len(time_steps)) + " points)")
	ax.set_ylabel('# ODE function calls')

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	fig.tight_layout()
	fig.savefig(dirname + "ode_func_calls_interval_length_{}_{}".format(experimentID, itr) + ".pdf")
	plt.close(fig)



	##########################
	# Number of ODE function eval based on number of time points to PREDICT
	fig = plt.figure(figsize=(5, 5), facecolor='white')
	ax = fig.add_subplot(111, frameon=False)

	for traj_id in range(20):
		one_traj = data[traj_id].unsqueeze(0)

		d = {"n_calls": [], "n_points": []}
		n_tp_to_choose = np.linspace(5, len(time_steps), n_measurements).astype(int)
		for i in n_tp_to_choose:
			idx = sorted(np.random.choice(len(time_steps), i, replace = False))
			time_steps_to_predict = time_steps[idx]
			if time_steps_to_predict[0] != 0:
				time_steps_to_predict = torch.cat((time_steps[:1], time_steps[idx]),0)

			observed_data = one_traj
			observed_time_steps = time_steps

			reconstructions, info = model.get_reconstruction(
				time_steps_to_predict, observed_data, observed_time_steps,
				mask = torch.ones_like(observed_data).to(get_device(data)), 
				n_traj_samples = 10)
			reconstructions = reconstructions.squeeze(1)

			d["n_calls"].append(info["n_calls"]) 
			d["n_points"].append(i) 
		
		ax.plot(d["n_points"], d["n_calls"], marker='o')
	
	ax.set_ylim(0, len(time_steps)+3)
	# plot the diagonal line
	# low_x, high_x = ax.get_xlim()
	# low_y, high_y = ax.get_ylim()
	# low = max(low_x, low_y)
	# high = min(high_x, high_y)
	# ax.plot([low,high], [low,high], ls="--", c=".3")

	ax.set_xlabel('# reconstructed time points')
	ax.set_ylabel('# ODE function calls')

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	fig.tight_layout()
	fig.savefig(dirname + "ode_func_calls_n_predicted_tp_{}_{}".format(experimentID, itr) + ".pdf")
	plt.close(fig)










def plot_metric_versus_n_points(model, test_dict, args,
	metric = "accuracy", experimentID = 0., itr = 0.):
	# Get the encoding of y(t0) using only observations t0 ... ti
	# Reconstruct the data from y(t0)
	# Look at the variance over y(t0) and over the decoded trajectory

	init_fonts()

	n_measurements = 20

	width = 5
	height = 5

	##########################
	# Number of ODE function eval based on number of time points to PREDICT
	fig = plt.figure(figsize=(5, 5), facecolor='white')
	ax = fig.add_subplot(111, frameon=False)

	d = {metric: [], "n_points": []}

	time_steps = test_dict["observed_tp"]
	n_tp_to_choose = np.linspace(5, len(time_steps), n_measurements).astype(int)
	for i in n_tp_to_choose:
		subsampled_dict = utils.subsample_observed_data(test_dict, n_tp_to_sample = i)

		# subsampled_dict["observed_data"] = subsampled_dict["observed_data"][:10]
		# subsampled_dict["observed_mask"] = subsampled_dict["observed_mask"][:10]
		# subsampled_dict["data_to_predict"] = subsampled_dict["data_to_predict"][:10]
		# subsampled_dict["mask_predicted_data"] = subsampled_dict["mask_predicted_data"][:10]
		# subsampled_dict["labels"] = subsampled_dict["labels"][:10]

		test_res = compute_loss_all_batches(model, subsampled_dict, args, experimentID = experimentID)

		assert(metric in test_res)
		d[metric].append(test_res[metric]) 
		d["n_points"].append(i) 
	

	pickle_file = "plots/" + str(experimentID) + \
		"/" + metric + "_vs_n_points_" + str(experimentID) + ".pickle"
	utils.dump_pickle(d, pickle_file)

	ax.plot(d["n_points"], d[metric], marker='o')
	# plot the diagonal line
	# low_x, high_x = ax.get_xlim()
	# low_y, high_y = ax.get_ylim()
	# low = max(low_x, low_y)
	# high = min(high_x, high_y)
	# ax.plot([low,high], [low,high], ls="--", c=".3")

	ax.set_xlabel('# reconstructed time points')
	ax.set_ylabel(metric)

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	fig.tight_layout()
	fig.savefig(dirname + metric + "_vs_n_points_{}".format(experimentID) + ".pdf")
	plt.close(fig)




def vizualize_mujoco(model, dataset_obj, data_dict, experimentID):
	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	print("Visualizing Mujoco...")
	dirname='hopper_imgs/{}/'.format(experimentID)
	
	# Condition on the first half of the sequence (first 100 points)
	# reconstruct the second half (100 points)
	n_tp_cond = len(time_steps)//2

	n_traj_samples = 1
	reconstr, info = model.get_reconstruction(time_steps, 
		observed_data, observed_time_steps, mask = observed_mask,
		n_traj_samples = n_traj_samples)

	num_traj_to_show = 20
	for i in range(min(num_traj_to_show, data.size(0))):
		print("Traj {} ...".format(i))

		# Visualize the true trajectory
		plot_truth = 'true_traj_{}_{}'.format(i, experimentID)
		dataset_obj.visualize(data[i].cpu(), plot_name = plot_truth, dirname = dirname)

		# Visualize reconstructions
		plot_rec = 'reconstr_traj_{}_{}'.format(i, experimentID)
		dataset_obj.visualize(reconstr[0,i].cpu(), dirname = dirname, plot_name = plot_rec)


class Visualizations2():
	def __init__(self, device):
		self.init_visualization()
		self.device = device

	def init_visualization(self):
		self.fig = plt.figure(figsize=(24, 20), facecolor='white')
		
		n_rows = 4
		self.ax_true_traj = self.fig.add_subplot(n_rows,4,1, frameon=False)
		self.ax_gen_traj = self.fig.add_subplot(n_rows,4,2, frameon=False)
		self.ax_traj_comparison = self.fig.add_subplot(n_rows,4,3, frameon=False)
		
		self.ax_y0 = self.fig.add_subplot(n_rows,4,4, frameon=False)	
		#self.ax_pred_density1 = self.fig.add_subplot(4,4,4, frameon=False)	
		# self.ax_true_density = self.fig.add_subplot(434, frameon=False)
		# self.ax_sampled_traj_density = self.fig.add_subplot(435, frameon=False)
		
		self.ax_kl = self.fig.add_subplot(n_rows,4,5, frameon=False)
		self.ax_loss = self.fig.add_subplot(n_rows,4,6, frameon=False)
		self.ax_pred_density3 = self.fig.add_subplot(n_rows,4,7, frameon=False)


		self.ax_pred_density1 = self.fig.add_subplot(n_rows,4,8, frameon=False)	
		#self.ax_pred_density2 = self.fig.add_subplot(4,4,8, frameon=False)	
		self.ax_y0_traj = self.fig.add_subplot(n_rows,4,9, frameon=False)
		self.ax_samples_same_traj = self.fig.add_subplot(n_rows,4,10, frameon=False)
		self.ax_traj_from_prior = self.fig.add_subplot(n_rows,4,11, frameon=False)

		self.ax_pred_density2 = self.fig.add_subplot(n_rows,4,12, frameon=False)	

		self.ax_calls = self.fig.add_subplot(n_rows,4,13, frameon=False)
		self.ax_extrapolation = self.fig.add_subplot(n_rows,4,14, frameon=False)
		self.ax_latent_traj = self.fig.add_subplot(n_rows,4,15, frameon=False)
		self.ax_latent_traj2 = self.fig.add_subplot(n_rows,4,16, frameon=False)

		# self.ax_extrapolation = self.fig.add_subplot(5,4,18, frameon=False)
		# self.ax_latent_traj = self.fig.add_subplot(5,4,19,frameon=False)
		# self.ax_latent_traj2 = self.fig.add_subplot(5,4,20, frameon=False)

		plt.show(block=False)



	def init_viz_for_all_density_plots(self):
		width = 12
		height = 8
		self.fig_density = plt.figure(figsize=(width, height), facecolor='white')
		
		self.ax_density1 = self.fig_density.add_subplot(2,3,1, frameon=False)
		self.ax_density2 = self.fig_density.add_subplot(2,3,2, frameon=False)
		self.ax_density3 = self.fig_density.add_subplot(2,3,3, frameon=False)
		self.ax_density4 = self.fig_density.add_subplot(2,3,4, frameon=False)		
		self.ax_density5 = self.fig_density.add_subplot(2,3,5, frameon=False)	
		self.ax_density6 = self.fig_density.add_subplot(2,3,6, frameon=False)	

		plt.show(block=False)



	def init_viz_for_one_density_plot(self):
		width = 4
		height = 4
		self.fig_density = plt.figure(figsize=(width, height), facecolor='white')
		
		# plot only part of the plot with true posterior and q distribution
		self.ax_density = self.fig_density.add_subplot(1,1,1, frameon=False)
		plt.show(block=False)


	def draw_one_density_plot(self, model, data_dict, experimentID, 
		itr, dirname, log_scale = False, multiply_by_poisson = False):
		
		scale = 4
		# cmap = add_white(plt.cm.get_cmap('Blues', 9)) # plt.cm.BuGn_r
		# cmap2 = add_white(plt.cm.get_cmap('Reds', 9)) # plt.cm.BuGn_r
		cmap = plt.cm.get_cmap('viridis')

		data = data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]

		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		npts = 50
		xx, yy, y0_grid = get_meshgrid(npts = npts, int_y1 = (-scale,scale), int_y2 = (-scale,scale))
		y0_grid = y0_grid.to(get_device(data))

		if model.use_poisson_proc:
			n_traj, n_dims = y0_grid.size()
			# append a vector of zeros to compute the integral of lambda and also zeros for the first point of lambda
			zeros = torch.zeros([n_traj, model.input_dim + model.latent_dim]).to(get_device(data))
			y0_grid_aug = torch.cat((y0_grid, zeros), -1)
		else:
			y0_grid_aug = y0_grid

		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y, extra_info = model.diffeq_solver(y0_grid_aug.unsqueeze(0), time_steps)
		
		if model.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = model.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
			
			assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
			assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		pred_x = model.decoder(sol_y)

		# Create density videos for 5 trajectories
		for traj_id in range(20):
			one_traj = data[traj_id]
			mask_one_traj = None
			if mask is not None:
				mask_one_traj = mask[traj_id].unsqueeze(0)
				mask_one_traj = mask_one_traj.repeat(npts**2,1,1).unsqueeze(0)

			self.ax_density.cla()

			# Plot: prior
			prior_density_grid = model.y0_prior.log_prob(y0_grid.unsqueeze(0)).squeeze(0)
			# Sum the density over two dimensions
			prior_density_grid = torch.sum(prior_density_grid, -1)
			
			# =================================================
			# Plot: p(x | y(t0))

			masked_gaussian_log_density_grid = masked_gaussian_log_density(pred_x, 
				one_traj.repeat(npts**2,1,1).unsqueeze(0),
				mask = mask_one_traj, 
				obsrv_std = model.obsrv_std).squeeze(-1)

			# Plot p(t | y(t0))
			if model.use_poisson_proc:
				poisson_info = {}
				poisson_info["int_lambda"] = int_lambda[:,:,-1,:]
				poisson_info["log_lambda_y"] = log_lambda_y
		
				poisson_log_density_grid = compute_poisson_proc_likelihood(
					one_traj.repeat(npts**2,1,1).unsqueeze(0),
					pred_x, poisson_info, mask = mask_one_traj)
				poisson_log_density_grid = poisson_log_density_grid.squeeze(0)
				
			# =================================================
			# Plot: p(x , y(t0))

			log_joint_density = prior_density_grid + masked_gaussian_log_density_grid
			if multiply_by_poisson:
				log_joint_density = log_joint_density + poisson_log_density_grid

			if log_scale:
				density_grid = log_joint_density
			else:
				density_grid = torch.exp(log_joint_density)

			density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
			density_grid = density_grid.cpu().numpy()

			self.ax_density.contourf(xx, yy, density_grid, cmap=cmap, alpha=1)

			# =================================================
			# Plot: q(y(t0)| x)
			#self.ax_density.set_title("Red: q(y(t0) | x)    Blue: p(x, y(t0))")
			self.ax_density.set_xlabel('y1(t0)')
			self.ax_density.set_ylabel('y2(t0)')

			data_w_mask = observed_data[traj_id].unsqueeze(0)
			if observed_mask is not None:
				data_w_mask = torch.cat((data_w_mask, observed_mask[traj_id].unsqueeze(0)), -1)
			y0_mu, y0_std = model.encoder_y0(
				data_w_mask, observed_time_steps)

			if model.use_poisson_proc:
				y0_mu = y0_mu[:, :, :model.latent_dim]
				y0_std = y0_std[:, :, :model.latent_dim]

			q_y0 = Normal(y0_mu, y0_std)

			q_density_grid = q_y0.log_prob(y0_grid)
			# Sum the density over two dimensions
			q_density_grid = torch.sum(q_density_grid, -1)
			if log_scale:
				density_grid = q_density_grid
			else:
				density_grid = torch.exp(q_density_grid)

			density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
			density_grid = density_grid.cpu().numpy()

			#self.ax_density.contourf(xx, yy, density_grid, cmap=cmap2, alpha=0.3)
			
			# =================================================
			
			self.ax_density.axis('off')
			os.makedirs(dirname, exist_ok=True)

			self.fig_density.tight_layout()
			plot_name = "y0_density_traj_"
			if multiply_by_poisson:
				plot_name += "poisson_"
			self.fig_density.savefig(dirname + plot_name + "{}_{}_{:03d}".format(traj_id, experimentID, itr) + ".pdf")




	def draw_all_density_plots(self, model, data_dict, experimentID, itr, dirname, log_scale = False):
		scale = 3
		cmap = add_white(plt.cm.get_cmap('Blues', 9)) # plt.cm.BuGn_r
		cmap2 = add_white(plt.cm.get_cmap('Reds', 9)) # plt.cm.BuGn_r

		data = data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]

		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		npts = 50
		xx, yy, y0_grid = get_meshgrid(npts = npts, int_y1 = (-scale,scale), int_y2 = (-scale,scale))
		y0_grid = y0_grid.to(get_device(data))

		if model.use_poisson_proc:
			n_traj, n_dims = y0_grid.size()
			# append a vector of zeros to compute the integral of lambda and also zeros for the first point of lambda
			zeros = torch.zeros([n_traj, model.input_dim + model.latent_dim]).to(get_device(data))
			y0_grid_aug = torch.cat((y0_grid, zeros), -1)
		else:
			y0_grid_aug = y0_grid

		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y, extra_info = model.diffeq_solver(y0_grid_aug.unsqueeze(0), time_steps)
		
		if model.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = model.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
			
			assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
			assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		pred_x = model.decoder(sol_y)

		# Create density videos for 5 trajectories
		for traj_id in range(10):
			one_traj = data[traj_id]
			mask_one_traj = None
			if mask is not None:
				mask_one_traj = mask[traj_id].unsqueeze(0)
				mask_one_traj = mask_one_traj.repeat(npts**2,1,1).unsqueeze(0)

			self.ax_density1.cla()
			self.ax_density2.cla()
			self.ax_density3.cla()
			self.ax_density4.cla()
			self.ax_density5.cla()
			self.ax_density6.cla()

			# Plot1: prior

			self.ax_density1.set_title("p(y0)")
			self.ax_density1.set_xlabel('y1(t0)')
			self.ax_density1.set_ylabel('y2(t0)')

			prior_density_grid = model.y0_prior.log_prob(y0_grid.unsqueeze(0)).squeeze(0)
			# Sum the density over two dimensions
			prior_density_grid = torch.sum(prior_density_grid, -1)
			if log_scale:
				density_grid = prior_density_grid
			else:
				density_grid = torch.exp(prior_density_grid)

			density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
			density_grid = density_grid.cpu().numpy()
			self.ax_density1.contourf(xx, yy, density_grid, cmap=cmap, alpha=0.5)

			# =================================================
			# Plot2: p(x | y(t0))

			self.ax_density2.set_title("p(x|y(t0))")
			self.ax_density2.set_xlabel('y1(t0)')
			self.ax_density2.set_ylabel('y2(t0)')

			masked_gaussian_log_density_grid = masked_gaussian_log_density(pred_x, 
				one_traj.repeat(npts**2,1,1).unsqueeze(0),
				mask = mask_one_traj, 
				obsrv_std = model.obsrv_std).squeeze(-1)

			if log_scale:
				density_grid = masked_gaussian_log_density_grid
			else:
				density_grid = torch.exp(masked_gaussian_log_density_grid)

			density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
			density_grid = density_grid.cpu().numpy()
			self.ax_density2.contourf(xx, yy, density_grid, cmap=cmap, alpha=0.5)

			# =================================================
			# Plot3: p(x , y(t0))
			self.ax_density3.set_title("p(x, y(t0))")
			self.ax_density3.set_xlabel('y1(t0)')
			self.ax_density3.set_ylabel('y2(t0)')

			log_joint_density = prior_density_grid + masked_gaussian_log_density_grid
			if log_scale:
				density_grid = log_joint_density
			else:
				density_grid = torch.exp(log_joint_density)

			density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
			density_grid = density_grid.cpu().numpy()

			self.ax_density3.contourf(xx, yy, density_grid, cmap=cmap, alpha=0.5)
			self.ax_density5.contourf(xx, yy, density_grid, cmap=cmap, alpha=0.7)

			# =================================================
			# Plot4: q(y(t0)| x)

			self.ax_density4.set_title("q(y(t0) | x)")
			self.ax_density4.set_xlabel('y1(t0)')
			self.ax_density4.set_ylabel('y2(t0)')

			self.ax_density5.set_title("Red: q(y(t0) | x)    Blue: p(x, y(t0))")
			self.ax_density5.set_xlabel('y1(t0)')
			self.ax_density5.set_ylabel('y2(t0)')

			data_w_mask = observed_data[traj_id].unsqueeze(0)
			if observed_mask is not None:
				data_w_mask = torch.cat((data_w_mask, observed_mask[traj_id].unsqueeze(0)), -1)
			y0_mu, y0_std = model.encoder_y0(
				data_w_mask, observed_time_steps)

			if model.use_poisson_proc:
				y0_mu = y0_mu[:, :, :model.latent_dim]
				y0_std = y0_std[:, :, :model.latent_dim]

			q_y0 = Normal(y0_mu, y0_std)

			q_density_grid = q_y0.log_prob(y0_grid)
			# Sum the density over two dimensions
			q_density_grid = torch.sum(q_density_grid, -1)
			if log_scale:
				density_grid = q_density_grid
			else:
				density_grid = torch.exp(q_density_grid)

			density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
			density_grid = density_grid.cpu().numpy()

			self.ax_density4.contourf(xx, yy, density_grid, cmap=cmap2, alpha=0.5)
			self.ax_density5.contourf(xx, yy, density_grid, cmap=cmap2, alpha=0.5)
			# vmin = density_grid.min() + 0.01

			# interval = density_grid.max() - density_grid.min()
			# dmin = density_grid.min()
			#im.set_clim(dmin + interval * 0.2, dmin + interval * 0.8)
			# =================================================
			# Plot5: E_x [ q(y(t0)|x) ]
			self.ax_density6.set_title("E_x [ q(y(t0)|x) ]")
			self.ax_density6.set_xlabel('y1(t0)')
			self.ax_density6.set_ylabel('y2(t0)')

			data_w_mask = observed_data[traj_id].unsqueeze(0)
			if observed_mask is not None:
				data_w_mask = torch.cat((data_w_mask, observed_mask[traj_id].unsqueeze(0)), -1)
			y0_mu, y0_std = model.encoder_y0(
				data_w_mask, observed_time_steps)

			if model.use_poisson_proc:
				y0_mu = y0_mu[:, :, :model.latent_dim]
				y0_std = y0_std[:, :, :model.latent_dim]

			q_y0 = Normal(y0_mu, y0_std)

			q_density_grid = q_y0.log_prob(y0_grid.unsqueeze(1))

			# Sum the density over two dimensions
			q_density_grid = torch.sum(q_density_grid, -1)
			if log_scale:
				density_grid = q_density_grid
			else:
				density_grid = torch.exp(q_density_grid)
			# Mean over all paths x
			density_grid = torch.mean(density_grid, 1)

			density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
			density_grid = density_grid.cpu().numpy()
			self.ax_density6.contourf(xx, yy, density_grid, cmap=cmap2, alpha=0.5)

			os.makedirs(dirname, exist_ok=True)

			self.fig_density.tight_layout()
			self.fig_density.savefig(dirname + "y0_density_traj_{}_{}_{:03d}".format(traj_id, experimentID, itr) + ".png")

			# =================================================
			# Save the reconstructions from different y_t0 on the grid

			# width = 20
			# height = 15
			# n_per_row = 5

			# fig, ax_list = plt.subplots(n_per_row, n_per_row, sharex='col', sharey='row', figsize=(width, height), facecolor='white')

			# n_traj, n_tp, n_input_dims = data.size()
			# pred_x_reshaped = pred_x.reshape(npts, npts, n_tp, n_input_dims)
			# y0_grid_reshaped = y0_grid.reshape(npts, npts, 2) # Assuming that we have only 2 latent dimensiosn
			
			# # Downsample the y0_grid
			# offset = npts // (n_per_row-1)
			# pred_x_reshaped = pred_x_reshaped[::offset, ::offset]
			# y0_grid_reshaped = y0_grid_reshaped[::offset, ::offset]

			# for i in range(n_per_row):
			# 	for j in range(n_per_row):
			# 		current_y0 = np.round(y0_grid_reshaped[i,j].numpy(),2)
			# 		# revert the rows so that y(t0) increases from left to right and from bottom up
			# 		ax = ax_list[n_per_row-i-1, j]
					
			# 		plot_trajectories(ax, one_traj.unsqueeze(0), time_steps, marker='o', linestyle='')
			# 		plot_trajectories(ax, pred_x_reshaped[i,j].unsqueeze(0), time_steps, 
			# 			title="y0: [" + str(current_y0[0]) + ";" + str(current_y0[1]) + "]", 
			# 			marker='', add_to_plot=True)

			# dirname = "plots/y0_density_plots/" + str(experimentID) + "/"
			# os.makedirs(dirname, exist_ok=True)

			# fig.tight_layout()
			# fig.savefig(dirname + "y0_grid_reconstr_" + str(experimentID) + '_traj_{}_{:03d}'.format(traj_id, itr) + ".png")
			# plt.close(fig)

			# =================================================
			# Sanity check: sample from true posterior

			# def plot_samples_from_true_posterior(log_joint_density, y0_grid, npts, experimentID, traj_id, time_steps):
			# 	# to avoid numerical issues				
			# 	log_joint_density = log_joint_density - torch.max(log_joint_density)

			# 	joint_density = torch.exp(log_joint_density)
			# 	joint_density = joint_density.reshape(npts, npts)

			# 	# normalize the distribution
			# 	joint_density = joint_density / torch.sum(joint_density)

			# 	n_samples = 5
			# 	sample_id = Categorical(joint_density.reshape(-1)).sample((n_samples,))
			# 	y_t0_true_posterior = y0_grid.reshape(npts**2, 2)[sample_id]
				
			# 	sol_y, extra_info = model.diffeq_solver(y_t0_true_posterior.unsqueeze(0), time_steps)
			# 	pred_x_true_post = model.decoder(sol_y).squeeze(0)

			# 	# Plot the samples from true posterior
			# 	width = 4 * n_samples
			# 	height = 3

			# 	fig, ax_list = plt.subplots(1, n_samples, sharex='col', sharey='row', figsize=(width, height), facecolor='white')

			# 	for i in range(n_samples):
			# 			current_y0 = np.round(y_t0_true_posterior[i].numpy(),2)
			# 			plot_trajectories(ax_list[i], one_traj.unsqueeze(0), time_steps, marker='o', linestyle='')
			# 			plot_trajectories(ax_list[i], pred_x_true_post[i].unsqueeze(0), time_steps, 
			# 				title="y0 from p(y(t0)|x) : [" + str(current_y0[0]) + ";" + str(current_y0[1]) + "]", 
			# 				marker='', add_to_plot=True)

			# 	dirname = "plots/y0_density_plots/" + str(experimentID) + "/"
			# 	os.makedirs(dirname, exist_ok=True)

			# 	fig.tight_layout()
			# 	fig.savefig(dirname + "y0_true_post_reconstr_" + str(experimentID) + '_traj_{}'.format(traj_id) + ".png")
			# 	plt.close(fig)

			# plot_samples_from_true_posterior(log_joint_density, y0_grid, npts, experimentID, traj_id, time_steps)



	def save_all_plots(self, data_dict, model,
		loss_list = None, kl_list = None, gp_param_list = None, y0_gp_param_list = None,
		call_count_d  = None, 
		plot_name = "", experimentID = 0., n_tp_to_sample = None):

		data =  data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]
		
		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		extrap_y = data
		time_steps_extrap = time_steps

		plot_dir = "plots/" + str(experimentID) + "/" + plot_name + "/"
		os.makedirs(plot_dir, exist_ok = True)
		
		n_dims = data.size()[-1]

		device = get_device(time_steps)
		# sample at the original time points
		time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 50).to(device)

		reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
			observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 10)
		#reconstructions = reconstructions.squeeze(0)

		# plot only 10 trajectories
		data_for_plotting = data[:10]
		reconstructions_for_plotting = info["pred_mean_y0"][:10]
		reconstr_std = reconstructions.std(dim=0)[:10]

		dim_to_show = 0
		max_y = max(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().max(),
			reconstructions[:,:,dim_to_show].cpu().numpy().max())
		min_y = min(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().min(),
			reconstructions[:,:,dim_to_show].cpu().numpy().min())

		# Sample from ground truth
		# true_traj_no_noise = dataset_object.sample_traj(
		# 	time_steps = time_steps, 
		# 	n_samples = 1, noise_weight = 0.)

		# # cut the dimension [:,:,0] which contains time stamps for the points -- they are all the same any way
		# true_traj_no_noise = true_traj_no_noise[:,:,1:].detach()
		# fitted_traj_no_noise, _ = model.get_reconstruction(time_steps_to_predict, true_traj_no_noise[:1], time_steps, )

		############################################
		# Plot truth and reconstruction
		save_all_dims(
			lambda ax, dim_to_show: plot_trajectories(ax, data_for_plotting, time_steps, 
			dim_to_show = dim_to_show, marker = 'o', linestyle=''),
			plot_file_name = plot_dir + "true_traj",
			title="True trajectories",
			n_dims = n_dims
		)

		def plot_func(ax, dim_to_show):
			plot_trajectories(ax, reconstructions_for_plotting, 
				time_steps_to_predict, dim_to_show = dim_to_show)
			plot_std(ax, reconstructions_for_plotting, reconstr_std, time_steps_to_predict, alpha=0.5)

		save_all_dims(
			plot_func,
			plot_file_name = plot_dir + "true_reconstr",
			title="Reconstructed trajectories",
			n_dims = n_dims
		)

		############################################
		# Density plots

		# save_all_dims(
		# 	lambda ax, dim_to_show: plot_estim_density_samples_prior(ax, model, time_steps_to_predict,  
		#  		min_y = min_y, max_y = max_y, dim_to_show = dim_to_show),
		# 	plot_file_name = plot_dir + "density_prior",
		# 	title = "Sample density E_p(z)[p(x|z)]",
		# 	n_dims = n_dims
		# )

		# save_all_dims(
		# 	lambda ax, dim_to_show: plot_estim_density_samples_q(ax, model.masked_gaussian_log_density, 
		# 		reconstructions_for_plotting, time_steps_to_predict,  
		# 		min_y = min_y, max_y = max_y, dim_to_show = dim_to_show),
		# 	plot_file_name = plot_dir + "density_approx_postrior",
		# 	title = "Sample density E_q(z| ground truth)[p(x|z)]",
		# 	n_dims = n_dims
		# )

		############################################
		# Noiseless trajectories

		# plot_trajectories(self.ax_traj_comparison, true_traj_no_noise, time_steps, 
		# 	min_y = min_y, max_y = max_y, label = "True mean trajectory")
		
		# plot_trajectories(self.ax_traj_comparison, fitted_traj_no_noise[0], time_steps_to_predict, 
		# 	min_y = min_y, max_y = max_y, add_to_plot = True, label = "Fitted mean trajectory",
		# 	title = "Mean trajectories (without noise)", add_legend = True)

		############################################
		# Plot loss and KL
		if loss_list is not None: 
			fig = plt.figure(figsize=(4, 4), facecolor='white')
			ax = fig.add_subplot(111, frameon=False)
			
			for key, val_list in loss_list.items():
				ax.plot(val_list, label = key)
			ax.set_title("Loss")
			ax.legend()
			fig.savefig(plot_dir + "loss.pdf")

		if kl_list is not None:
			fig = plt.figure(figsize=(4, 4), facecolor='white')
			ax = fig.add_subplot(111, frameon=False)

			ax.plot(kl_list, color = "r")
			ax.set_title("KL div")
			fig.savefig(plot_dir + "kl.pdf")

		############################################
		# Get several samples for the same trajectory
		gt_traj = data_for_plotting[:1]
		first_point = gt_traj[:,0]

		if len(time_steps_to_predict.size()) == 1:
			time_steps_to_predict_one = time_steps_to_predict
			time_steps_one = time_steps
		else:
			time_steps_to_predict_one = time_steps_to_predict[:,0]
			time_steps_one = time_steps[:,0]

		observed_mask_ = observed_mask[:1] if observed_mask is not None else None
		samples_same_traj, _ = model.get_reconstruction(time_steps_to_predict, 
			observed_data[:1], observed_time_steps, mask = observed_mask_, n_traj_samples = 5)
		samples_same_traj = samples_same_traj.squeeze(1)
		
		def plot_func(ax, dim_to_show):
			plot_trajectories(ax, samples_same_traj, time_steps_to_predict_one, dim_to_show = dim_to_show)
			plot_trajectories(ax, gt_traj, time_steps_one, linestyle = "dashed", 
				label = "True traj", add_to_plot = True, dim_to_show = dim_to_show) 

		save_all_dims(
			plot_func,
			plot_file_name = plot_dir + "samples_same_traj",
			title="Samples for the same trajectory",
			n_dims = n_dims
		)

		############################################
		# Plot trajectories from prior

		if isinstance(model, ODEVAE):
			if len(time_steps_to_predict.size()) == 1:
				time_steps_to_predict_one = time_steps_to_predict
			else:
				time_steps_to_predict_one = time_steps_to_predict[:,0]

			traj_from_prior = model.sample_traj_from_prior(time_steps_to_predict_one, n_traj_samples = 6)
			# Since in this case n_traj = 1, n_traj_samples -- requested number of samples from the prior, squeeze n_traj dimension
			traj_from_prior = traj_from_prior.squeeze(1)

			save_all_dims(
				lambda ax, dim_to_show: plot_trajectories(ax, traj_from_prior, time_steps_to_predict, dim_to_show = dim_to_show),
				plot_file_name = plot_dir + "traj_from_prior",
				title="Trajectories from prior",
				n_dims = n_dims
			)

			if call_count_d is not None:
				fig = plt.figure(figsize=(4, 4), facecolor='white')
				ax = fig.add_subplot(111, frameon=False)
				
				ax.scatter(call_count_d["n_tp"],  call_count_d["n_calls"], c = call_count_d["n_tp"],  
					s = 100, cmap='hsv', alpha=0.5, marker='o')
				ax.set_title("# ODE func calls versus # time points")
				ax.set_xlabel('# time points')
				ax.set_ylabel('# ODE func calls')

				fig.savefig(plot_dir + "ode_func_evals.pdf")

		############################################
		# Plot extrapolation to future points	
		# device = get_device(time_steps)
		# true_traj_extrapolation, time_steps_to_extrapolate = extrap_y, time_steps_extrap

		# num_traj_to_show = 1
		# if len(time_steps_to_predict.size()) == 1:
		# 	time_steps_to_extrapolate_one = time_steps_to_extrapolate
		# 	time_steps_one = time_steps
		# else:
		# 	time_steps_to_extrapolate_one = time_steps_to_extrapolate[:,0]
		# 	time_steps_one = time_steps[:,0]

		# samples_extrapolation, _ = model.get_reconstruction(time_steps_to_extrapolate_one, 
		# 	true_traj_extrapolation[:num_traj_to_show,:len(time_steps)], time_steps_one, n_traj_samples = 5)
		# samples_extrapolation = samples_extrapolation.squeeze(1)
		# true_traj_extrapolation = true_traj_extrapolation[:num_traj_to_show]

		# def plot_func(ax, dim_to_show):
		# 	plot_trajectories(ax, samples_extrapolation, time_steps_to_extrapolate, dim_to_show = dim_to_show)
		# 	plot_trajectories(ax, true_traj_extrapolation, time_steps_to_extrapolate, linestyle = "dashed", 
		# 		label = "True traj", add_to_plot = True, dim_to_show = dim_to_show)
		# 	ax.axvline(x = time_steps.max())

		# save_all_dims(
		# 	plot_func,
		# 	plot_file_name = plot_dir + "extrapolation",
		# 	title="Extrapolation (5 samples for for the same trajectory)",
		# 	n_dims = n_dims
		# )

		################################################
		# Plot trajectories in the latent space

		latent_traj1 = info["latent_traj"][0,:1]
		# swap the n_traj dimension (which is 1) and latent dimensions
		# i.e. display each dimension as a separate trajectory
		# shape after: [1, n_tp, n_latent_dims]
		latent_traj1 = latent_traj1.permute(2,1,0)

		if len(time_steps_to_predict.size()) == 1:
			time_steps_to_predict_one = time_steps_to_predict
		else:
			time_steps_to_predict_one = time_steps_to_predict[:,0]

		fig = plt.figure(figsize=(6, 4), facecolor='white')
		ax = fig.add_subplot(111, frameon=False)
		plot_trajectories(ax, latent_traj1, time_steps_to_predict_one, title="Latents for one trajectory")
		fig.savefig(plot_dir + "latents_traj1" + ".pdf")

		if info["latent_traj"].size(1) > 1:
			latent_traj2 = info["latent_traj"][0,1:2]
			latent_traj2 = latent_traj2.permute(2,1,0)

			fig = plt.figure(figsize=(6, 4), facecolor='white')
			ax = fig.add_subplot(111, frameon=False)
			plot_trajectories(ax, latent_traj2, time_steps_to_predict_one, title="Latents for one trajectory")
			fig.savefig(plot_dir + "latents_traj2" + ".pdf")

		################################################
		# Plot how GP parameter changes over training

		# if len(gp_param_list) > 0:
		# 	fig = plt.figure(figsize=(6, 4), facecolor='white')
		# 	ax = fig.add_subplot(111, frameon=False)

		# 	ax.plot(np.array(gp_param_list), color = "b")
		# 	ax.set_title("Parameter for GP for z(t)")
		# 	fig.savefig(plot_dir + "gp_parameter_zt" + ".pdf")

		# if len(y0_gp_param_list) > 0:
		# 	fig = plt.figure(figsize=(6, 4), facecolor='white')
		# 	ax = fig.add_subplot(111, frameon=False)

		# 	ax.plot(np.array(y0_gp_param_list), color = "b")
		# 	ax.set_title("Parameter for GP for y0")
		# 	fig.savefig(plot_dir + "gp_parameter_y0" + ".pdf")

		################################################

		if isinstance(model, ODEVAE):
			if isinstance(model.encoder_y0, Encoder_y0_ode_combine):
				# Plot ODEs going from different y_tis to y_t0
				

				print(observed_data.size())
				print(observed_time_steps.size())
				print(observed_mask.size())

				if observed_mask is not None:
					observed_data_w_mask = torch.cat((observed_data, observed_mask), -1)

				for traj_id in range(10):
					fig = plt.figure(figsize=(6, 4), facecolor='white')
					ax = fig.add_subplot(111, frameon=False)

					print("observed_data_w_mask")
					print(observed_data_w_mask.size())

					means_y0, var_y0 = model.encoder_y0(observed_data_w_mask[traj_id:(traj_id+1)], observed_time_steps, save_info = True)
					#means_y0 shape: [1, n_traj, n_latent_dims]
					plotting_info = model.encoder_y0.extra_info
					model.encoder_y0.extra_info = None


					for ode_traj in plotting_info:
						time_points = ode_traj["time_points"]
						t_i = time_points[-1]
						# The solution goes from t_i-1 to t_i
						ode_sol = ode_traj["ode_sol"]
						
						yi_ode = ode_traj["yi_ode"]
						yi = ode_traj["yi"]
						yi_std = ode_traj["yi_std"]

						# Take the first training example and first dimension
						ode_sol = ode_sol[0, :1, :, :1]
					
						if len(time_steps.size()) == 1:
							time_points_one = time_points
						else:
							time_points_one = time_points[:,0]
							t_i = t_i[0]

						plot_trajectories(ax, ode_sol, time_points_one, marker = '', add_to_plot = True)
						plot_std(ax, ode_sol, yi_std[0, 0, 0], time_points_one, add_to_plot = False)
							
						ax.scatter(t_i.cpu(), yi_ode[0, 0, 0].cpu(), marker='o', label = "y from ode")
						ax.scatter(t_i.cpu(), yi[0, 0, 0].cpu(), marker='x', label = "chosen y_i")
						#ax.set_title("Recognition model for y0")

						if "yi_from_data" in ode_traj:
							yi_from_data = ode_traj["yi_from_data"]
							ax.scatter(t_i.cpu(), yi_from_data[0, 0, 0].cpu(), marker="D", label = "y from data")
					
						fig.tight_layout()
						fig.savefig(plot_dir + "recog_model_traj_{}.pdf".format(traj_id))


			if isinstance(model.encoder_y0, Encoder_y0_from_rnn):
				# Plot ODEs going from different y_tis to y_t0
				means_y0, var_y0 = model.encoder_y0(observed_data, observed_time_steps,
					mask = observed_mask)
				#means_y0 shape: [1, n_traj, n_latent_dims]
				plotting_info = model.encoder_y0.extra_info

				time_points = plotting_info["time_points"]				
				# LSTM output shape: (seq_len, batch, num_directions * hidden_size)
				lstm_outputs = plotting_info["rnn_outputs"]
				# Take the first training example and first dimension
				lstm_outputs = lstm_outputs[:,:1,:].permute(2,0,1)
			
				if len(time_steps.size()) == 1:
					time_points_one = time_points
				else:
					time_points_one = time_points[:,0]

				fig = plt.figure(figsize=(6, 4), facecolor='white')
				ax = fig.add_subplot(111, frameon=False)
				plot_trajectories(ax, lstm_outputs, time_points_one,
					title="RNN hidden states", add_to_plot = True)

				fig.tight_layout()
				fig.savefig(plot_dir + "recog_model.pdf")



def make_predictions_w_samples_same_traj(model, experimentID, device, extrap = False):
	dataset_obj = Periodic_1d(
		init_freq = 1., init_amplitude = 0.7,
		final_amplitude = 1., final_freq = 0.7, 
		y0 = 1.)

	time_steps = torch.linspace(0, 5, 200)

	for noise_weight in [0., 0.01, 0.1, 0.5, 1.]:
		data = dataset_obj.sample_traj(time_steps, 
			n_samples = 1, noise_weight = noise_weight)
		data = data[:,:,1:]

		data_dict = {"dataset_obj": dataset_obj,
				"test_y": data.to(device), 
				"test_time_steps": time_steps.to(device)}

		if extrap:
			data_dict = utils.split_data_extrap(data_dict, "test")
		else:
			data_dict = utils.split_data_interp(data_dict, "test")

		for sample_tp in [20, 30, 50, 80]:
			data_dict_subsampled = utils.subsample_observed_data(data_dict, n_tp_to_sample = sample_tp)

			plot_reconstructions(model, data_dict_subsampled, experimentID = experimentID, 
				itr = "noise_{}_subsampled_{}".format(noise_weight, sample_tp), width = 10)


def plot_hopper_performance_plot():
	hopper_res_file = "results/results_hopper_interp_n_subsampled_points_likelihood.csv"
	res = pd.read_csv(hopper_res_file, sep=",")
	res = res[['model', '10', '20', '30', '50']]

	fig = plt.figure(figsize=(5, 5), facecolor='white')
	ax = fig.add_subplot(111, frameon=False)
	x_values = res.columns[1:].map(int)

	aureg = ['ode_gru_rnn', 'classic_rnn_cell_gru', 'classic_rnn_cell_expdecay', 'classic_rnn_cell_expdecay_input_decay']
	enc_dec = ['y0_rnn', 'rnn_vae_cell_gru', 'y0_ode_combine']

	for m in res['model']:
		if (m not in aureg) and (m not in enc_dec):
			print("The type of model " + m + " not found!")

	for i in range(res.shape[0]):
		model_name = res.iloc[i,0]
		linetype = 'solid' if (model_name in enc_dec) else 'dashed'
		# showing MSE, but the table reports negative MSE
		ax.plot(x_values, res.iloc[i,1:], marker='o', ls = linetype,
			label = model_name)

	ax.legend()
	fig.tight_layout()
	fig.savefig("plots/hopper_performance_{}".format(experimentID) + ".pdf")
	plt.close(fig)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Make demos 
# Video how the model fit changes with number  of points

def make_video_reconstruct_encoding_t0_ti(model, data_dict,
	experimentID, itr, n_traj_to_show = 10):

	init_fonts(MEDIUM_SIZE)

	data =  data_dict["data_to_predict"]
	time_steps = data_dict["tp_to_predict"]
	mask = data_dict["mask_predicted_data"]
	
	observed_data =  data_dict["observed_data"]
	observed_time_steps = data_dict["observed_tp"]
	observed_mask = data_dict["observed_mask"]

	dirname = "plots/" + str(experimentID) + "/"
	os.makedirs(dirname, exist_ok=True)

	time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 100)
	time_steps_to_predict = time_steps_to_predict.to(get_device(data))

	n_traj_to_show = min(n_traj_to_show, data.size(0))
	dim_to_show = 0

	for traj_id in range(n_traj_to_show):
		one_traj = data[traj_id, :, :].unsqueeze(0)
		one_observed_traj = observed_data[traj_id, :, :].unsqueeze(0)
		one_observed_mask = observed_mask[traj_id, :, :].unsqueeze(0)

		plot_name = dirname + "/reconstr_encoding_t0_ti_traj_{}_{}_{}".format(traj_id, experimentID, itr)

		non_missing_idx = np.where(one_observed_mask[0,:,dim_to_show].cpu().numpy())[0]
		non_missing_idx = non_missing_idx[5:]
		non_missing_idx = non_missing_idx[::2]
		def get_n_points(i):
			return i* 2 + 5

		for i, ti_index in enumerate(non_missing_idx):
			data_to_ti = one_observed_traj[:,:ti_index]
			mask_to_ti = one_observed_mask[:,:ti_index]
			ts_to_ti = observed_time_steps[:ti_index]

			reconstructions, info = model.get_reconstruction(
				time_steps_to_predict, data_to_ti, ts_to_ti,
				mask = mask_to_ti, n_traj_samples = 10)
			reconstructions = reconstructions.squeeze(1)

			fig = plt.figure(figsize=(8,5))
			ax = fig.add_subplot(111, frameon=False)

			plot_trajectories(ax, data_to_ti, ts_to_ti, mask = mask_to_ti, marker='o', linestyle='')
			plot_trajectories(ax, reconstructions, time_steps_to_predict, 
				title= str(get_n_points(i)) + " observed points", # Mean sq: " + str(mean_sq_error),  
				marker='', add_to_plot=True) # marker='o'

			ax.set_ylim(one_traj.cpu().min()-0.1, one_traj.cpu().max() + 0.1)
			fig.tight_layout()
			fig.savefig(plot_name + "_{:03d}".format(i) + ".png")
			plt.close()

		convert_to_movie(plot_name + "_%03d.png", plot_name + ".mp4", rate = 5)






