###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
from lib.latent_ode import LatentODE

from lib.likelihood_eval import masked_gaussian_log_density
try:
	import umap
except:
	print("Couldn't import umap")

from generate_timeseries import Periodic_1d
from person_activity import PersonActivity

from lib.utils import compute_loss_all_batches


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
LARGE_SIZE = 22

def init_fonts(main_font_size = LARGE_SIZE):
	plt.rc('font', size=main_font_size)          # controls default text sizes
	plt.rc('axes', titlesize=main_font_size)     # fontsize of the axes title
	plt.rc('axes', labelsize=main_font_size - 2)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=main_font_size - 2)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=main_font_size - 2)    # fontsize of the tick labels
	plt.rc('legend', fontsize=main_font_size - 2)    # legend fontsize
	plt.rc('figure', titlesize=main_font_size)  # fontsize of the figure title


def plot_trajectories(ax, traj, time_steps, min_y = None, max_y = None, title = "", 
		add_to_plot = False, label = None, add_legend = False, dim_to_show = 0,
		linestyle = '-', marker = 'o', mask = None, color = None, linewidth = 1):
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
		d = traj[i].cpu().numpy()[:, dim_to_show]
		ts = time_steps.cpu().numpy()
		if mask is not None:
			m = mask[i].cpu().numpy()[:, dim_to_show]
			d = d[m == 1]
			ts = ts[m == 1]
		ax.plot(ts, d, linestyle = linestyle, label = label, marker=marker, color = color, linewidth = linewidth)

	if add_legend:
		ax.legend()


def plot_std(ax, traj, traj_std, time_steps, min_y = None, max_y = None, title = "", 
	add_to_plot = False, label = None, alpha=0.2, color = None):

	# take only the first (and only?) dimension
	mean_minus_std = (traj - traj_std).cpu().numpy()[:, :, 0]
	mean_plus_std = (traj + traj_std).cpu().numpy()[:, :, 0]

	for i in range(traj.size()[0]):
		ax.fill_between(time_steps.cpu().numpy(), mean_minus_std[i], mean_plus_std[i], 
			alpha=alpha, color = color)



def plot_vector_field(ax, odefunc, latent_dim, device):
	# Code borrowed from https://github.com/rtqichen/ffjord/blob/29c016131b702b307ceb05c70c74c6e802bb8a44/diagnostics/viz_toy.py
	K = 13j
	y, x = np.mgrid[-6:6:K, -6:6:K]
	K = int(K.imag)
	zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
	if latent_dim > 2:
		# Plots dimensions 0 and 2
		zs = torch.cat((zs, torch.zeros(K * K, latent_dim-2)), 1)
	dydt = odefunc(0, zs)
	dydt = -dydt.cpu().detach().numpy()
	if latent_dim > 2:
		dydt = dydt[:,:2]

	mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
	dydt = (dydt / mag)
	dydt = dydt.reshape(K, K, 2)

	ax.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], #color = dydt[:, :, 0],
		cmap="coolwarm", linewidth=2)

	# ax.quiver(
	# 	x, y, dydt[:, :, 0], dydt[:, :, 1],
	# 	np.exp(logmag), cmap="coolwarm", pivot="mid", scale = 100,
	# )
	ax.set_xlim(-6, 6)
	ax.set_ylim(-6, 6)
	#ax.axis("off")



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


class Visualizations():
	def __init__(self, device):
		self.init_visualization()
		init_fonts(SMALL_SIZE)
		self.device = device

	def init_visualization(self):
		self.fig = plt.figure(figsize=(12, 7), facecolor='white')
		
		self.ax_traj = []
		for i in range(1,4):
			self.ax_traj.append(self.fig.add_subplot(2,3,i, frameon=False))

		# self.ax_density = []
		# for i in range(4,7):
		# 	self.ax_density.append(self.fig.add_subplot(3,3,i, frameon=False))

		#self.ax_samples_same_traj = self.fig.add_subplot(3,3,7, frameon=False)
		self.ax_latent_traj = self.fig.add_subplot(2,3,4, frameon=False)
		self.ax_vector_field = self.fig.add_subplot(2,3,5, frameon=False)
		self.ax_traj_from_prior = self.fig.add_subplot(2,3,6, frameon=False)

		self.plot_limits = {}
		plt.show(block=False)

	def set_plot_lims(self, ax, name):
		if name not in self.plot_limits:
			self.plot_limits[name] = (ax.get_xlim(), ax.get_ylim())
			return

		xlim, ylim = self.plot_limits[name]
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

	def draw_one_density_plot(self, ax, model, data_dict, traj_id, 
		multiply_by_poisson = False):
		
		scale = 5
		cmap = add_white(plt.cm.get_cmap('Blues', 9)) # plt.cm.BuGn_r
		cmap2 = add_white(plt.cm.get_cmap('Reds', 9)) # plt.cm.BuGn_r
		#cmap = plt.cm.get_cmap('viridis')

		data = data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]

		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		npts = 50
		xx, yy, z0_grid = get_meshgrid(npts = npts, int_y1 = (-scale,scale), int_y2 = (-scale,scale))
		z0_grid = z0_grid.to(get_device(data))

		if model.latent_dim > 2:
			z0_grid = torch.cat((z0_grid, torch.zeros(z0_grid.size(0), model.latent_dim-2)), 1)

		if model.use_poisson_proc:
			n_traj, n_dims = z0_grid.size()
			# append a vector of zeros to compute the integral of lambda and also zeros for the first point of lambda
			zeros = torch.zeros([n_traj, model.input_dim + model.latent_dim]).to(get_device(data))
			z0_grid_aug = torch.cat((z0_grid, zeros), -1)
		else:
			z0_grid_aug = z0_grid

		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = model.diffeq_solver(z0_grid_aug.unsqueeze(0), time_steps)
		
		if model.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = model.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
			
			assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
			assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		pred_x = model.decoder(sol_y)

		# Plot density for one trajectory
		one_traj = data[traj_id]
		mask_one_traj = None
		if mask is not None:
			mask_one_traj = mask[traj_id].unsqueeze(0)
			mask_one_traj = mask_one_traj.repeat(npts**2,1,1).unsqueeze(0)

		ax.cla()

		# Plot: prior
		prior_density_grid = model.z0_prior.log_prob(z0_grid.unsqueeze(0)).squeeze(0)
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

		density_grid = torch.exp(log_joint_density)

		density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
		density_grid = density_grid.cpu().numpy()

		ax.contourf(xx, yy, density_grid, cmap=cmap, alpha=1)

		# =================================================
		# Plot: q(y(t0)| x)
		#self.ax_density.set_title("Red: q(y(t0) | x)    Blue: p(x, y(t0))")
		ax.set_xlabel('z1(t0)')
		ax.set_ylabel('z2(t0)')

		data_w_mask = observed_data[traj_id].unsqueeze(0)
		if observed_mask is not None:
			data_w_mask = torch.cat((data_w_mask, observed_mask[traj_id].unsqueeze(0)), -1)
		z0_mu, z0_std = model.encoder_z0(
			data_w_mask, observed_time_steps)

		if model.use_poisson_proc:
			z0_mu = z0_mu[:, :, :model.latent_dim]
			z0_std = z0_std[:, :, :model.latent_dim]

		q_z0 = Normal(z0_mu, z0_std)

		q_density_grid = q_z0.log_prob(z0_grid)
		# Sum the density over two dimensions
		q_density_grid = torch.sum(q_density_grid, -1)
		density_grid = torch.exp(q_density_grid)

		density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
		density_grid = density_grid.cpu().numpy()

		ax.contourf(xx, yy, density_grid, cmap=cmap2, alpha=0.3)
	


	def draw_all_plots_one_dim(self, data_dict, model,
		plot_name = "", save = False, experimentID = 0.):

		data =  data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]
		
		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		device = get_device(time_steps)

		time_steps_to_predict = time_steps
		if isinstance(model, LatentODE):
			# sample at the original time points
			time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 100).to(device)

		reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
			observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 10)

		n_traj_to_show = 3
		# plot only 10 trajectories
		data_for_plotting = observed_data[:n_traj_to_show]
		mask_for_plotting = observed_mask[:n_traj_to_show]
		reconstructions_for_plotting = reconstructions.mean(dim=0)[:n_traj_to_show]
		reconstr_std = reconstructions.std(dim=0)[:n_traj_to_show]

		dim_to_show = 0
		max_y = max(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().max(),
			reconstructions[:,:,dim_to_show].cpu().numpy().max())
		min_y = min(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().min(),
			reconstructions[:,:,dim_to_show].cpu().numpy().min())

		############################################
		# Plot reconstructions, true postrior and approximate posterior

		cmap = plt.cm.get_cmap('Set1')
		for traj_id in range(3):
			# Plot observations
			plot_trajectories(self.ax_traj[traj_id], 
				data_for_plotting[traj_id].unsqueeze(0), observed_time_steps, 
				mask = mask_for_plotting[traj_id].unsqueeze(0),
				min_y = min_y, max_y = max_y, #title="True trajectories", 
				marker = 'o', linestyle='', dim_to_show = dim_to_show,
				color = cmap(2))
			# Plot reconstructions
			plot_trajectories(self.ax_traj[traj_id],
				reconstructions_for_plotting[traj_id].unsqueeze(0), time_steps_to_predict, 
				min_y = min_y, max_y = max_y, title="Sample {} (data space)".format(traj_id), dim_to_show = dim_to_show,
				add_to_plot = True, marker = '', color = cmap(3), linewidth = 3)
			# Plot variance estimated over multiple samples from approx posterior
			plot_std(self.ax_traj[traj_id], 
				reconstructions_for_plotting[traj_id].unsqueeze(0), reconstr_std[traj_id].unsqueeze(0), 
				time_steps_to_predict, alpha=0.5, color = cmap(3))
			self.set_plot_lims(self.ax_traj[traj_id], "traj_" + str(traj_id))
			
			# Plot true posterior and approximate posterior
			# self.draw_one_density_plot(self.ax_density[traj_id],
			# 	model, data_dict, traj_id = traj_id,
			# 	multiply_by_poisson = False)
			# self.set_plot_lims(self.ax_density[traj_id], "density_" + str(traj_id))
			# self.ax_density[traj_id].set_title("Sample {}: p(z0) and q(z0 | x)".format(traj_id))
		############################################
		# Get several samples for the same trajectory
		# one_traj = data_for_plotting[:1]
		# first_point = one_traj[:,0]

		# samples_same_traj, _ = model.get_reconstruction(time_steps_to_predict, 
		# 	observed_data[:1], observed_time_steps, mask = observed_mask[:1], n_traj_samples = 5)
		# samples_same_traj = samples_same_traj.squeeze(1)
		
		# plot_trajectories(self.ax_samples_same_traj, samples_same_traj, time_steps_to_predict, marker = '')
		# plot_trajectories(self.ax_samples_same_traj, one_traj, time_steps, linestyle = "", 
		# 	label = "True traj", add_to_plot = True, title="Reconstructions for the same trajectory (data space)")

		############################################
		# Plot trajectories from prior
		
		if isinstance(model, LatentODE):
			torch.manual_seed(1991)
			np.random.seed(1991)

			traj_from_prior = model.sample_traj_from_prior(time_steps_to_predict, n_traj_samples = 3)
			# Since in this case n_traj = 1, n_traj_samples -- requested number of samples from the prior, squeeze n_traj dimension
			traj_from_prior = traj_from_prior.squeeze(1)

			plot_trajectories(self.ax_traj_from_prior, traj_from_prior, time_steps_to_predict, 
				marker = '', linewidth = 3)
			self.ax_traj_from_prior.set_title("Samples from prior (data space)", pad = 20)
			#self.set_plot_lims(self.ax_traj_from_prior, "traj_from_prior")
		################################################

		# Plot z0
		# first_point_mu, first_point_std, first_point_enc = info["first_point"]

		# dim1 = 0
		# dim2 = 1
		# self.ax_z0.cla()
		# # first_point_enc shape: [1, n_traj, n_dims]
		# self.ax_z0.scatter(first_point_enc.cpu()[0,:,dim1], first_point_enc.cpu()[0,:,dim2])
		# self.ax_z0.set_title("Encodings z0 of all test trajectories (latent space)")
		# self.ax_z0.set_xlabel('dim {}'.format(dim1))
		# self.ax_z0.set_ylabel('dim {}'.format(dim2))

		################################################
		# Show vector field
		self.ax_vector_field.cla()
		plot_vector_field(self.ax_vector_field, model.diffeq_solver.ode_func, model.latent_dim, device)
		self.ax_vector_field.set_title("Slice of vector field (latent space)", pad = 20)
		self.set_plot_lims(self.ax_vector_field, "vector_field")
		#self.ax_vector_field.set_ylim((-0.5, 1.5))

		################################################
		# Plot trajectories in the latent space

		# shape before [1, n_traj, n_tp, n_latent_dims]
		# Take only the first sample from approx posterior
		latent_traj = info["latent_traj"][0,:n_traj_to_show]
		# shape before permute: [1, n_tp, n_latent_dims]

		self.ax_latent_traj.cla()
		cmap = plt.cm.get_cmap('Accent')
		n_latent_dims = latent_traj.size(-1)

		custom_labels = {}
		for i in range(n_latent_dims):
			col = cmap(i)
			plot_trajectories(self.ax_latent_traj, latent_traj, time_steps_to_predict, 
				title="Latent trajectories z(t) (latent space)", dim_to_show = i, color = col, 
				marker = '', add_to_plot = True,
				linewidth = 3)
			custom_labels['dim ' + str(i)] = Line2D([0], [0], color=col)
		
		self.ax_latent_traj.set_ylabel("z")
		self.ax_latent_traj.set_title("Latent trajectories z(t) (latent space)", pad = 20)
		self.ax_latent_traj.legend(custom_labels.values(), custom_labels.keys(), loc = 'lower left')
		self.set_plot_lims(self.ax_latent_traj, "latent_traj")

		################################################

		self.fig.tight_layout()
		plt.draw()

		if save:
			dirname = "plots/" + str(experimentID) + "/"
			os.makedirs(dirname, exist_ok=True)
			self.fig.savefig(dirname + plot_name)





