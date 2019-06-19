import time
import numpy as np

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import lib.utils as utils

def plot_gaussian_process():
	#Test sampling from a gaussian process
	fig = plt.figure( facecolor='white')
	ax = fig.add_subplot(111, frameon=False)

	for i in range(3):
		gp = GaussianProcess(0.1, kernel = "WienerProcess")

		vals = []
		ts = np.linspace(0, 1., 100)
		for t in ts:
			vals.append(gp.sample([t])[0])
		ax.plot(ts[1:], vals[1:])
	plt.show()


class WhiteNoise():
	def __init__(self, sigma):
		self.sigma = sigma
		self.samples = []
		self.sample_ts = []

	def clear(self):
		self.samples = []
		self.sample_ts = []

	def sample_multidim(self, time_points, dim = 1):
		"""
		Sample from Write Noise (uncorrelated gaussian)
		time_points: time points to generate samples for

		dim: optional dimensionality of sampled GP

		Returns:
			np.array with GP samples of shape [len(time_points), dim]
		"""

		sample_ = np.random.normal(scale = self.sigma, size = [len(time_points), dim])
		self.samples.extend(sample_)
		self.sample_ts.extend(time_points)
		return sample_


class GaussianProcess():
	def __init__(self, sigma, kernel = "intWienerProcess"):
		self.sigma = sigma
		self.samples = []
		self.sample_ts = []

		if kernel == "intWienerProcess":
			self.kernel = self.intWienerProcess
		elif kernel == "WienerProcess":
			self.kernel = self.WienerProcess
		elif kernel == "rbf_kernel":
			self.kernel = self.rbf_kernel
		elif kernel == "whiteNoise":
			self.kernel = self.whiteNoise
		elif callable(kernel):
			self.kernel = kernel
		else:
			raise Exception("Unknown kernel {}".format(kernel))

	def clear(self):
		self.samples = []
		self.sample_ts = []

	def intWienerProcess(self, t1, t2):
		t1 =  np.array(t1).reshape(-1,1)
		t2 = np.array(t2).reshape(-1,1)

		t1_tiled = np.tile(t1, len(t2))
		t2_tiled = np.tile(t2, len(t1)).T

		m = np.amin(np.concatenate((np.expand_dims(t1_tiled,2), np.expand_dims(t2_tiled,2)),2),2)
		kernel = self.sigma**2 * (m**3 / 3 + np.multiply(np.abs(t1_tiled-t2_tiled), m**2) / 2)
		return kernel

	def WienerProcess(self, t1, t2):
		t1 =  np.array(t1).reshape(-1,1)
		t2 = np.array(t2).reshape(-1,1)

		t1_tiled = np.tile(t1, len(t2))
		t2_tiled = np.tile(t2, len(t1)).T

		m = np.amin(np.concatenate((np.expand_dims(t1_tiled,2), np.expand_dims(t2_tiled,2)),2),2)
		kernel = self.sigma**2 * m
		return kernel

	def whiteNoise(self, t1, t2):
		t1 =  np.array(t1).reshape(-1,1)
		t2 = np.array(t2).reshape(-1,1)
		assert(len(t1) == len(t2))
		return np.diag([self.sigma] * len(t1))

	def rbf_kernel(self, a, b, param = 0.1):
		a =  np.array(a).reshape(-1,1)
		b = np.array(b).reshape(-1,1)

		sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
		return np.exp(-.5 * (1/param) * sqdist)

	def get_mean_and_cov(self, time_steps):
		mu = mean = np.repeat([0.], len(time_steps))
		cov_matrix = self.kernel(time_steps, time_steps)
		return mu, cov_matrix

	def sample(self, time_points):
		def find_ind(list, elem):
			ind = np.argwhere(np.array(self.sample_ts) == elem)
			if len(ind) != 0:
				return ind[0,0]
			return -1

		# if some of the time_points have been sampled before,
		# just find them in the self.samples list
		# if the time point is not there, add it to ts_to_sample and sample!
		samples = []
		ts_to_sample = []
		for t in time_points:
			ind = find_ind(self.sample_ts, t)

			if ind != -1:
				samples.append(self.samples[ind])
			else:
				samples.append(-1)
				ts_to_sample.append(t)

		new_samples = []
		if len(self.samples) == 0:
			cov_matrix = self.kernel(ts_to_sample, ts_to_sample)
			sample_ = np.random.multivariate_normal(mean = np.repeat([0.], len(ts_to_sample)) , cov = cov_matrix)

			self.samples.extend(sample_)
			self.sample_ts.extend(ts_to_sample)
			new_samples = sample_
		elif len(ts_to_sample) != 0:
			cov_matrix = self.kernel(self.sample_ts, self.sample_ts)
			L = np.linalg.cholesky(cov_matrix + 0.00005*np.eye(len(self.sample_ts)))
			K_ss = self.kernel(ts_to_sample, ts_to_sample)

			# Compute the mean at our test points.
			K_s = self.kernel(self.sample_ts, ts_to_sample)

			Lk = np.linalg.solve(L, K_s)
			mu = np.dot(Lk.T, np.linalg.solve(L, self.samples)).reshape((-1,))

			# Compute the standard deviation so we can plot it
			s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
			stdv = np.sqrt(s2)

			# Draw samples from the posterior at our test points.
			L = np.linalg.cholesky(K_ss + 1e-6*np.eye(1) - np.dot(Lk.T, Lk))
			f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(len(ts_to_sample),1)))
			f_post = f_post[:,0]

			self.samples.extend(f_post)
			self.sample_ts.extend(ts_to_sample)
			new_samples = f_post

		samples = np.array(samples).astype(float)
		new_samples = np.array(new_samples)
		samples[np.where(np.array(samples) == -1.)[0]] = new_samples

		return np.array(samples)

	def sample_multidim(self, time_points, dim = 1):
		"""
		Sample from Gaussian Process
		time_points: time points to generate samples for

		dim: optional dimensionality of sampled GP

		Returns:
			np.array with GP samples of shape [len(time_points), dim]
		"""

		def find_ind(list, elem):
			ind = np.argwhere(np.array(self.sample_ts) == elem)
			if len(ind) != 0:
				return ind[0,0]
			return -1

		# if some of the time_points have been sampled before,
		# just find them in the self.samples list
		# if the time point is not there, add it to ts_to_sample and sample!
		samples = np.zeros((len(time_points), dim))

		ts_to_sample = []
		mask_points_to_sample = np.zeros((len(time_points)))

		for i, t in enumerate(time_points):
			# check if we have already sampled this time point
			ind = find_ind(self.sample_ts, t)

			if ind != -1:
				# if we have already sampled at t, just append the previous sample
				samples[i] = self.samples[ind]
			else:
				# if we have not sampled at t, append "-1" for now
				mask_points_to_sample[i] = 1
				ts_to_sample.append(t)

		new_samples = []
		if len(self.samples) == 0:
			# we need to samples all time points -- none of them have been sampled before
			shape = 1
			if dim is not None:
				shape = dim

			cov_matrix = self.kernel(ts_to_sample, ts_to_sample) + 0.00005*np.eye(len(ts_to_sample))
			sample_ = np.random.multivariate_normal(mean = np.repeat([0.], len(ts_to_sample)) , cov = cov_matrix, size = shape)
			sample_ = np.transpose(sample_)

			new_samples = sample_
			self.sample_ts.extend(ts_to_sample)
			if len(self.samples) != 0:
				self.samples = np.concatenate((self.samples, sample_), 0)
			else:
				self.samples = sample_
		elif len(ts_to_sample) != 0:
			# Sample new points for time points ts_to_sample
			cov_matrix = self.kernel(self.sample_ts, self.sample_ts)
			L = np.linalg.cholesky(cov_matrix + 0.00005*np.eye(len(self.sample_ts)))
			K_ss = self.kernel(ts_to_sample, ts_to_sample)

			# Compute the mean at our test points.
			K_s = self.kernel(self.sample_ts, ts_to_sample)

			Lk = np.linalg.solve(L, K_s)
			mu = np.dot(Lk.T, np.linalg.solve(L, self.samples))

			# Compute the standard deviation so we can plot it
			s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
			stdv = np.sqrt(s2)

			# Draw samples from the posterior at our test points.
			L = np.linalg.cholesky(K_ss + 1e-6*np.eye(1) - np.dot(Lk.T, Lk))
			f_post = mu + np.dot(L, np.random.normal(size=(len(ts_to_sample),dim)))

			self.samples = np.concatenate((self.samples, f_post), 0)

			self.sample_ts.extend(ts_to_sample)
			new_samples = f_post

		samples = np.array(samples).astype(float)
		new_samples = np.array(new_samples)

		if sum(mask_points_to_sample) != 0:
			# fill in samples for t's which we haven't sampled before
			samples[np.where(mask_points_to_sample == 1)[0]] = new_samples

		assert(self.samples.shape[0] == len(self.sample_ts))
		return np.array(samples)


def make_time_point_pairs(t1, t2):
	t1 = t1.view(-1,1)
	t2 = t2.view(-1,1)

	t1_tiled = t1.repeat(1, len(t2))
	t2_tiled = t2.repeat(1, len(t1))
	t2_tiled = t2_tiled.transpose(0,1)

	t1_tiled = t1_tiled.unsqueeze(2)
	t2_tiled = t2_tiled.unsqueeze(2)

	concat = torch.cat((t1_tiled, t2_tiled), 2)
	return concat


class PytorchGaussianProcess(nn.Module):
	def __init__(self, kernel_param, sigma_obs, kernel = None, device = torch.device("cpu")):
		super(PytorchGaussianProcess, self).__init__()

		self.kernel_param = kernel_param
		self.sigma_obs = sigma_obs
		self.device = device

		if callable(kernel):
			self.kernel = kernel
		elif kernel == "WienerProcess":
			self.kernel = self.WienerProcess
		elif kernel == "RBF":
			self.kernel = self.rbf_kernel
		elif kernel == "OU":
			self.kernel = self.OrnsteinUhlenbeck
		else:
			raise Exception("Unknown kernel {}".format(kernel))

	def rbf_kernel(self, a, b):
		a = a.view(-1,1)
		b = b.view(-1,1)

		sqdist = torch.sum(a**2,1).view(-1,1) + torch.sum(b**2,1) - 2*torch.mm(a, torch.transpose(b, 0,1))
		return torch.exp(-.5 * torch.abs(self.kernel_param) * sqdist)

	def WienerProcess(self, t1, t2):
		concat = make_time_point_pairs(t1, t2)
		m = torch.min(concat, 2)[0]

		kernel = self.kernel_param**2 * m
		return kernel


	def OrnsteinUhlenbeck(self, t1, t2):
		concat = make_time_point_pairs(t1, t2)
		# k(t, t') = exp( |t - t'| / scale
		kernel = torch.exp(-(concat[:,:,0] - concat[:,:,1]).abs() * torch.abs(self.kernel_param))
		return kernel


	def get_mean_and_cov(self, time_steps):
		mu = torch.Tensor().new_zeros((1,), device = self.device).repeat(len(time_steps))
		cov_matrix = self.kernel(time_steps, time_steps)
		return mu, cov_matrix

	def gp_regression(self, time_steps, prev_tp = None, prev_samples = None):
		"""
		time_steps: time poits for which we want to get mean and covarince from the GP
		n_gp_samples: number of samples for each time point. Used only when we don't condition on previous samples
		prev_samples: points on which we want to condition GP. Requires 2d shape: [n_time_points, -1]
		prev_tp: time stamps for prev_samples
		"""
		cov_xx = self.kernel(prev_tp, prev_tp) + torch.eye(len(prev_tp)).to(self.device) * (self.sigma_obs**2 + 1e-6)

		cov_x_star_x = self.kernel(time_steps, prev_tp)
		cov_x_x_star = torch.transpose(cov_x_star_x, 0, 1)
		cov_x_star_x_star = self.kernel(time_steps, time_steps)

		chol_xx = torch.cholesky(cov_xx, upper = True)

		tmp = differentiable_potrs(prev_samples, chol_xx)  # NOTE: this assume zero-mean. We'll want to learn the mean later on.
		mean = torch.mm(cov_x_star_x, tmp)
		cov_matrix = cov_x_star_x_star - torch.mm(cov_x_star_x, differentiable_potrs(cov_x_x_star, chol_xx))

		return mean, cov_matrix

	def density(self, samples, time_steps, prev_tp = None, prev_samples = None):
		mean, cov_matrix = self.gp_regression(time_steps, prev_tp, prev_samples)
		gaussian = MultivariateNormal(mean, covariance_matrix = cov_matrix)
		return gaussian.log_prob(samples)

	def sample(self, mean, cov_matrix):
		L = torch.potrf(cov_matrix + torch.eye(cov_matrix.size(0)) * 1e-3, upper = False)
		d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(self.device), torch.Tensor([1.]).to(self.device))
		r = d.sample(mean.size()).squeeze(-1)
		gp_sample = torch.mm(L.float(),r) + mean.float()
		return gp_sample

	def get_mean_var_for_every_point(self, time_steps_to_predict, prev_tp, prev_samples):
		# get mean and variance for each point individually instead of covariance matrix
		# Useful for plotting variance across the timeline
		means_flat, stds = [], []
		for t in time_steps_to_predict:
			m, c = self.gp_regression(t, 
				prev_tp = prev_tp, prev_samples = prev_samples)
			means_flat.append(m)
			stds.append(c)

		means_flat = torch.stack(means_flat)
		stds = torch.stack(stds)
		return means_flat, stds


	def do_regression_from_ind_points(self, tp_for_eval, ind_points, ind_points_ts):
		# shape before: [n_samples, n_tp, n_dim]
		# shape after: [n_tp, n_samples, n_dim]
		ind_points_flat = ind_points.permute(1,0,2)
		ind_points_flat = utils.flatten(ind_points_flat.contiguous(),1)

		means, cov_matrix = self.gp_regression(tp_for_eval, 
			prev_tp = ind_points_ts, prev_samples = ind_points_flat)

		# shape: [n_gp_samples * n_samples * n_dims, n_tp]
		means_flat = means.transpose(0,1)
		cov_matrix_flat = cov_matrix.repeat(means_flat.size()[0],1,1)
		return means_flat, cov_matrix_flat



def differentiable_potrs(b, U):
	return torch.trtrs(torch.trtrs(b, U.transpose(0,1), upper=False)[0], U, upper=True)[0]

if __name__ == "__main__":
	plot_gaussian_process()
