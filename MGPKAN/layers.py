
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F

from .abstr import Layer
from .kernels import SquaredExponentialKernel
from .funcs import WeightFunc, InducValueFunc



class FCLayer(Layer):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		num_comp: int = 1,
		weighted: bool = False,
		degree: int = 0,
	):
		Layer.__init__(self, in_dim, out_dim)
		
		self.num_induc = num_induc  # M
		self.num_comp = num_comp  # K
		# partition ~ [D, Q, K-1]
		self.init_partition()
		# center ~ [D, Q, K]
		self._center = Parameter(torch.zeros(out_dim, in_dim, num_comp))
		# kernel ~ [D, Q, K]
		self.kernel = SquaredExponentialKernel([out_dim, in_dim, num_comp])
		# z ~ [D, Q, K, M]
		self._induc_loc = Parameter(torch.rand(out_dim, in_dim, num_comp, num_induc).mul(2).sub(1))
		# induc_noise ~ [D, Q, K]
		self._induc_noise = Parameter(torch.ones(out_dim, in_dim, num_comp).mul(1e-2).exp().sub(1).log())
		# induc_value ~ [P+1, D, Q, K, M]
		self.induc_value_func = InducValueFunc([out_dim, in_dim, num_comp], num_induc, degree)
		# weight ~ [D, Q, K]
		self.weight_func = WeightFunc([out_dim, in_dim, num_comp]) if weighted else None
	
	def init_partition(self) -> None:
		_partition = torch.ones(self.out_dim, self.in_dim, self.num_comp-1)
		_partition = _partition.div(torch.tensor(range(self.num_comp,1,-1))).reciprocal().sub(1).log().mul(-1)
		self._partition = Parameter(_partition)

	def transform_partition(self) -> None:
		# [D, Q, K+1]
		_partition = self._partition.sigmoid()
		partition = torch.zeros(self.out_dim, self.in_dim, self.num_comp+1)
		p = torch.ones([]).mul(-1)
		for k in range(self.num_comp-1):
			partition[...,k+1] = p = p.mul(_partition[...,k].mul(-1).add(1)).add(_partition[...,k])
		partition[...,0] = -1
		partition[...,-1] = 1
		self.partition = partition

	@property
	def center(self) -> Tensor:
		# [D, Q, K]
		l = self.partition[...,:-1]
		u = self.partition[...,1:]
		return u.sub(l).mul(self._center.sigmoid()).add(l)
	
	def transform_induc_loc(self) -> None:
		# [D, Q, K, M]
		l = self.partition[...,:-1].unsqueeze(-1)
		u = self.partition[...,1:].unsqueeze(-1)
		self.induc_loc = u.sub(l).mul(self._induc_loc.sigmoid()).add(l)

	@property
	def induc_noise(self) -> Tensor:
		return F.softplus(self._induc_noise) + 1e-6
	
	def weighted_input(
		self, x_mean: Tensor, x_var: Optional[Tensor] = None,
	) -> tuple[Tensor, Tensor, Union[Tensor, None]]:
		"""
		x_mean & x_var ~ [B, Q]
		->
		q_mean & q_var & weight ~ [B, D, Q, K]
		"""
		self.transform_partition()
		if self.weight_func is None:
			q_mean = x_mean.unsqueeze(1).unsqueeze(-1)
			q_var = None if x_var is None else x_var.unsqueeze(1).unsqueeze(-1)
			weight = torch.ones([self.num_comp]).div(self.num_comp)
		else:
			q_mean, q_var, weight = self.weight_func(self.center, x_mean, x_var)
		return q_mean, q_var, weight
	
	def induc_value(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> Tensor:
		_, q_mean, q_var = self.weighted_input(x_mean, x_var)
		self.transform_induc_loc()
		self.kernel.transform_lengthscale()
		return self.induc_value_func(self.induc_loc, self.kernel.lengthscale, q_mean, q_var)
	
	def unmixed_w(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Union[Tensor, None]]:
		"""
		x_mean & x_var ~ [B, Q]
		->
		w_mean & w_var & weight ~ [B, D, Q, K]
		"""
		# q_mean & q_var & weight ~ [B, D, Q, K]
		q_mean, q_var, weight = self.weighted_input(x_mean, x_var)
		self.transform_induc_loc()
		self.kernel.transform_lengthscale()
		# Kuu_inv ~ [D, Q, M, M]
		Kuu_inv = self.kernel.Kuu_inv(self.induc_loc, self.induc_noise)
		# Cuf ~ [B, D, Q, M]
		Cuf = self.kernel.Cuf(self.induc_loc, q_mean, q_var)
		# Cff ~ [B, D, Q]
		Cff = self.kernel.Cff(q_var)
		# E[u(x)] ~ [B, D, Q, K, M]
		induc_value = self.induc_value_func(self.induc_loc, self.kernel.lengthscale, q_mean, q_var)
		w_mean = Kuu_inv.matmul(induc_value.unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)
		w_var = Cff.sub(Kuu_inv.matmul(Cuf.unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)).mul(self.kernel.outputscale)
		return w_mean, w_var, weight

	def compute_w(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
		"""
		x_mean & x_var ~ [B, Q]
		->
		w_mean & w_var ~ [B, D, Q]
		"""
		w_mean, w_var, weight = self.unmixed_w(x_mean, x_var)
		w_mean = w_mean.mul(weight).sum(-1)
		w_var = w_var.mul(weight.pow(2)).sum(-1)
		return w_mean, w_var

	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
		"""
		x_mean & x_var ~ [B, Q]
		->
		f_mean & f_var ~ [B, D]
		"""
		w_mean, w_var = self.compute_w(x_mean, x_var)
		f_mean = w_mean.sum(-1)
		f_var = w_var.sum(-1) + 1e-6
		return f_mean, f_var

	def extra_repr(self) -> str:
		return (
			f"in_dim={self.in_dim}, out_dim={self.out_dim}\n"
			+f"num_induc={self.num_induc}, num_comp={self.num_comp}, degree={self.induc_value_func.degree}"
		)


class NormLayer(Layer):

	def __init__(self):
		Layer.__init__(self)

	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Union[Tensor, None]]:
		return x_mean.tanh(), None if x_var is None else x_var.tanh()


