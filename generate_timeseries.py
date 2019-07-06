###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# Create a synthetic dataset
from __future__ import absolute_import, division
from __future__ import print_function
import os
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
import lib.utils as utils

# ======================================================================================

def get_next_val(init, t, tmin, tmax, final = None):
	if final is None:
		return init
	val = init + (final - init) / (tmax - tmin) * t
	return val


def generate_periodic(time_steps, init_freq, init_amplitude, starting_point, 
	final_freq = None, final_amplitude = None, phi_offset = 0.):

	tmin = time_steps.min()
	tmax = time_steps.max()

	data = []
	t_prev = time_steps[0]
	phi = phi_offset
	for t in time_steps:
		dt = t - t_prev
		amp = get_next_val(init_amplitude, t, tmin, tmax, final_amplitude)
		freq = get_next_val(init_freq, t, tmin, tmax, final_freq)
		phi = phi + 2 * np.pi * freq * dt # integrate to get phase

		y = amp * np.sin(phi) + starting_point
		t_prev = t
		data.append([t,y])
	return np.array(data)

def assign_value_or_sample(value, sampling_interval = [0.,1.]):
	if value is None:
		int_length = sampling_interval[1] - sampling_interval[0]
		return np.random.random() * int_length + sampling_interval[0]
	else:
		return value

class TimeSeries:
	def __init__(self, device = torch.device("cpu")):
		self.device = device
		self.z0 = None

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 4), facecolor='white')
		self.ax = self.fig.add_subplot(111, frameon=False)
		plt.show(block=False)

	def visualize(self, truth):
		self.ax.plot(truth[:,0], truth[:,1])

	def add_noise(self, traj_list, time_steps, noise_weight):
		n_samples = traj_list.size(0)

		# Add noise to all the points except the first point
		n_tp = len(time_steps) - 1
		noise = np.random.sample((n_samples, n_tp))
		noise = torch.Tensor(noise).to(self.device)

		traj_list_w_noise = traj_list.clone()
		# Dimension [:,:,0] is a time dimension -- do not add noise to that
		traj_list_w_noise[:,1:,0] += noise_weight * noise
		return traj_list_w_noise



class Periodic_1d(TimeSeries):
	def __init__(self, device = torch.device("cpu"), 
		init_freq = 0.3, init_amplitude = 1.,
		final_amplitude = 10., final_freq = 1., 
		z0 = 0.):
		"""
		If some of the parameters (init_freq, init_amplitude, final_amplitude, final_freq) is not provided, it is randomly sampled.
		For now, all the time series share the time points and the starting point.
		"""
		super(Periodic_1d, self).__init__(device)
		
		self.init_freq = init_freq
		self.init_amplitude = init_amplitude
		self.final_amplitude = final_amplitude
		self.final_freq = final_freq
		self.z0 = z0

	def sample_traj(self, time_steps, n_samples = 1, noise_weight = 1.,
		cut_out_section = None):
		"""
		Sample periodic functions. 
		"""
		traj_list = []
		for i in range(n_samples):
			init_freq = assign_value_or_sample(self.init_freq, [0.4,0.8])
			if self.final_freq is None:
				final_freq = init_freq
			else:
				final_freq = assign_value_or_sample(self.final_freq, [0.4,0.8])
			init_amplitude = assign_value_or_sample(self.init_amplitude, [0.,1.])
			final_amplitude = assign_value_or_sample(self.final_amplitude, [0.,1.])

			noisy_z0 = self.z0 + np.random.normal(loc=0., scale=0.1)

			traj = generate_periodic(time_steps, init_freq = init_freq, 
				init_amplitude = init_amplitude, starting_point = noisy_z0, 
				final_amplitude = final_amplitude, final_freq = final_freq)

			# Cut the time dimension
			traj = np.expand_dims(traj[:,1:], 0)
			traj_list.append(traj)

		# shape: [n_samples, n_timesteps, 2]
		# traj_list[:,:,0] -- time stamps
		# traj_list[:,:,1] -- values at the time stamps
		traj_list = np.array(traj_list)
		traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
		traj_list = traj_list.squeeze(1)

		traj_list = self.add_noise(traj_list, time_steps, noise_weight)
		return traj_list

