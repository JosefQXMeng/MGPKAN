
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.linalg import cholesky_ex, solve_triangular
from torch.nn import Module, Parameter
from torch.nn import functional as F

from .utils import SquaredExponentialExpectation



class Kernel(Module, ABC):

	def __init__(self, size: list[int]):
		Module.__init__(self)

		# [D, Q, K]
		self.size = size
		self._outputscale = Parameter(torch.zeros(size))
		self._lengthscale = Parameter(torch.zeros(size))

	@property
	def outputscale(self) -> Tensor:
		return F.softplus(self._outputscale) + 1e-6

	@property
	def lengthscale(self) -> Tensor:
		return F.softplus(self._lengthscale) + 1e-6
	
	@abstractmethod
	def expectation(self) -> Tensor:
		...

	def Kuu_cholesky(self, induc_loc: Tensor) -> Tensor:
		"""
		z ~ [D, Q, K, M]
		->
		Kuu = L @ L.mT ~ [D, Q, K, M, M]
		"""
		lengthscale = self.lengthscale.unsqueeze(-1).unsqueeze(-1)
		dist = induc_loc.unsqueeze(-1).sub(induc_loc.unsqueeze(-2))
		Kuu = self.expectation(lengthscale, dist)
		return cholesky_ex(Kuu + 1e-6 * torch.eye(Kuu.size(-1))).L

	def Kuu_inv(self, induc_loc: Tensor) -> Tensor:
		"""
		z ~ [D, Q, K, M]
		->
		Kuu_inv ~ [D, Q, K, M, M]
		"""
		L = self.Kuu_cholesky(induc_loc)
		L_inv = solve_triangular(L, torch.eye(induc_loc.size(-1)), upper=False)
		return L_inv.mT.matmul(L_inv)

	def Cuf(
		self, induc_loc: Tensor, x_mean: Tensor, x_var: Optional[Tensor] = None,
	) -> Tensor:
		"""
		z ~ [D, Q, K, M]
		x_mean & x_var ~ [B, Q, C]
		->
		Cuf ~ [B, D, Q, K, M]
		"""
		lengthscale = self.lengthscale.unsqueeze(-1).unsqueeze(-1)
		dist = x_mean.unsqueeze(1).unsqueeze(-2).unsqueeze(-1).sub(induc_loc.unsqueeze(-2))
		if x_var is not None:
			x_var = x_var.unsqueeze(1).unsqueeze(-2).unsqueeze(-1)
		return self.expectation(lengthscale, dist, x_var).mean(-2)

	def Cff(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> Tensor:
		"""
		x_mean & x_var ~ [B, Q, C]
		->
		Cff ~ [B, D, Q, K]
		"""
		lengthscale = self.lengthscale.unsqueeze(-1).unsqueeze(-1)
		x_mean = x_mean.unsqueeze(1).unsqueeze(-2)
		if x_var is not None:
			x_var = x_var.unsqueeze(1).unsqueeze(-2)
		mu = x_mean.unsqueeze(-1).sub(x_mean.unsqueeze(-2))
		sigma_sq = None if x_var is None else x_var.unsqueeze(-1).add(x_var.unsqueeze(-2))
		return self.expectation(lengthscale, mu, sigma_sq).mean([-1,-2])


class SquaredExponentialKernel(Kernel):

	def __init__(self, size: list[int]):
		Kernel.__init__(self, size)

	def expectation(
		self, lengthscale: Tensor, mu: Optional[Tensor] = None, sigma_sq: Optional[Tensor] = None,
	) -> Tensor:
		return SquaredExponentialExpectation(lengthscale, mu, sigma_sq)


