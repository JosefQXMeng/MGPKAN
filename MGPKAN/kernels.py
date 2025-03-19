
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
		self._outputscale = Parameter(torch.ones(size).exp().sub(1).log())
		self._lengthscale = Parameter(torch.ones(size).div(size[-1]**2).exp().sub(1).log())

	@property
	def outputscale(self) -> Tensor:
		return F.softplus(self._outputscale) + 1e-6

	def transform_lengthscale(self) -> None:
		self.lengthscale = F.softplus(self._lengthscale) + 1e-6
	
	@abstractmethod
	def expectation(self) -> Tensor:
		...

	def Kuu_inv(self, induc_loc: Tensor, induc_noise: Tensor) -> Tensor:
		"""
		z ~ [D, Q, K, M]
		induc_noise ~ [D, Q, K]
		->
		Kuu_inv ~ [D, Q, K, M, M]
		"""
		mu = induc_loc.unsqueeze(-1).sub(induc_loc.unsqueeze(-2))
		Kuu = self.expectation(self.lengthscale.unsqueeze(-1).unsqueeze(-1), mu)
		induc_noise = induc_noise.unsqueeze(-1).unsqueeze(-1).mul(torch.eye(Kuu.size(-1)))
		L = cholesky_ex(Kuu.add(induc_noise)).L
		L_inv = solve_triangular(L, torch.eye(L.size(-1)), upper=False)
		return L_inv.mT.matmul(L_inv)

	def Cuf(self, induc_loc: Tensor, q_mean: Tensor, q_var: Optional[Tensor] = None) -> Tensor:
		"""
		z ~ [D, Q, K, M]
		q_mean & q_var ~ [B, D, Q, K]
		->
		Cuf ~ [B, D, Q, K, M]
		"""
		mu = q_mean.unsqueeze(-1).sub(induc_loc)
		sigma_sq = None if q_var is None else q_var.unsqueeze(-1)
		return self.expectation(self.lengthscale.unsqueeze(-1), mu, sigma_sq)

	def Cff(self, q_var: Optional[Tensor] = None) -> Tensor:
		"""
		q_var ~ [B, D, Q, K]
		->
		Cff ~ [B, D, Q, K]
		"""
		sigma_sq = None if q_var is None else q_var.mul(2)
		return self.expectation(self.lengthscale, None, sigma_sq)


class SquaredExponentialKernel(Kernel):

	def __init__(self, size: list[int]):
		Kernel.__init__(self, size)

	def expectation(self, lengthscale: Tensor, mu: Optional[Tensor] = None, sigma_sq: Optional[Tensor] = None):
		return SquaredExponentialExpectation(lengthscale, mu, sigma_sq)


