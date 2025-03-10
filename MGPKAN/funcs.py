
import math
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import functional as F

from .utils import SquaredExponentialExpectation



class InducValueFunc(Module):

	def __init__(self, size: list[int], num_induc: int, degree: int):
		Module.__init__(self)

		self.degree = degree  # P
		self.induc_value = Parameter(torch.zeros(degree+1, *size, num_induc))
		self.init_moment_coef(degree)

	def init_moment_coef(self, degree: int) -> None:
		coef_list = []
		for n in range(1, degree+1):
			sublist = []
			for k in range(n+1):
				if not k % 2:
					sublist.append(math.comb(n, k) * math.prod(range(1, k, 2)))
			coef_list.append(sublist)
		self.moment_coef = coef_list

	def compute_moment(self, mean: Tensor, var: Tensor, n: int) -> Tensor:
		assert n in range(1, self.degree+1)
		E = torch.zeros([])
		for k in range(n+1):
			if not k % 2:
				E = mean.pow(n-k).mul(var.pow(k//2)).mul(self.moment_coef[n-1][k//2]).add(E)
		return E

	def compute_density(
		self, lengthscale: Tensor, induc_loc: Tensor, q_mean: Tensor, q_var: Optional[Tensor] = None,
	) -> tuple[Tensor, Union[Tensor, None]]:
		"""
		q_mean & q_var ~ [B, D, Q, K]
		->
		g_mean & g_var ~ [B, D, Q, K, M]
		"""
		q_mean = q_mean.unsqueeze(-1)
		if q_var is not None:
			q_var = q_var.unsqueeze(-1)
			lengthscale = lengthscale.unsqueeze(-1)
			cross_prod = q_mean.mul(lengthscale).add(q_var.mul(induc_loc))
			var_sum = q_var.add(lengthscale)
		g_mean = q_mean if q_var is None else cross_prod.div(var_sum)
		g_var = torch.zeros([]) if q_var is None else q_var.mul(lengthscale).div(var_sum)
		return g_mean, g_var
	
	def forward(self, lengthscale: Tensor, induc_loc: Tensor, q_mean: Tensor, q_var: Optional[Tensor] = None) -> Tensor:
		"""
		q_mean & q_var ~ [B, D, Q, K]
		->
		E[u(x)] ~ [B, D, D, K, M]
		"""
		E_u = self.induc_value[0]
		if self.degree:
			g_mean, g_var = self.compute_density(induc_loc, lengthscale, q_mean, q_var)
			for p in range(1, self.degree+1):
				E_u = self.compute_moment(g_mean, g_var, p).mul(self.induc_value[p]).add(E_u)
		return E_u


class WeightFunc(Module):

	def __init__(self, size: list[int]):
		Module.__init__(self)

		# [D, Q, K]
		self._outputscale = Parameter(torch.ones(size).exp().sub(1).log())
		self._lengthscale = Parameter(torch.ones(size).div(size[-1]**2).exp().sub(1).log())
	
	@property
	def outputscale(self) -> Tensor:
		return F.softplus(self._outputscale) + 1e-6
	
	@property
	def lengthscale(self) -> Tensor:
		return F.softplus(self._lengthscale) + 1e-6

	def forward(
		self, center: Tensor, x_mean: Tensor, x_var: Optional[Tensor] = None,
	) -> tuple[Tensor, Tensor, Union[Tensor, None]]:
		"""
		c ~ [D, Q, K]
		x_mean & x_var ~ [B, Q]
		->
		w & q_mean & q_var ~ [B, D, Q, K]
		"""
		lengthscale = self.lengthscale
		x_mean = x_mean.unsqueeze(1).unsqueeze(-1)
		if x_var is not None:
			x_var = x_var.unsqueeze(1).unsqueeze(-1)
			cross_prod = x_mean.mul(lengthscale).add(x_var.mul(center))
			var_sum = x_var.add(lengthscale)
		weight = SquaredExponentialExpectation(lengthscale, x_mean.sub(center), x_var).mul(self.outputscale)
		weight = weight.div(weight.sum(-1, keepdim=True))
		q_mean = x_mean if x_var is None else cross_prod.div(var_sum)
		q_var = None if x_var is None else x_var.mul(lengthscale).div(var_sum)
		return weight, q_mean, q_var


