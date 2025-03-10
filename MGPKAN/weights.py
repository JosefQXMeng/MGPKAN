
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import functional as F

from .utils import SquaredExponentialExpectation



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


