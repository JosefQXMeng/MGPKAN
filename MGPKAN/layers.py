
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from .abstr import Layer
from .kernels import SquaredExponentialKernel



class FCLayer(Layer):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		num_comp: int = 1,
	):
		Layer.__init__(self, in_dim, out_dim)
		
		self.num_induc = num_induc  # M
		self.num_comp = num_comp  # K

		self.kernel = SquaredExponentialKernel([out_dim, in_dim, num_comp])
		# z ~ [K, D, Q, M]
		self.induc_loc = Parameter(torch.rand(out_dim, in_dim, num_comp, num_induc).mul(2).sub(1))
		self.polynomial_coef = Parameter(torch.zeros(out_dim, in_dim, num_comp, num_induc))

	def compute_w(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Union[Tensor, None]]:
		"""
		x_mean & x_var ~ [B, Q, C]
		->
		w_mean & w_var ~ [B, D, Q, K]
		"""

		# Kuu_inv ~ [D, Q, K, M, M]
		Kuu_inv = self.kernel.Kuu_inv(self.induc_loc)
		# Cuf ~ [B, D, Q, K, M]
		Cuf = self.kernel.Cuf(self.induc_loc, x_mean, x_var)
		# Cff ~ [(B, D, Q, K)]
		Cff = self.kernel.Cff(x_mean, x_var)

		w_mean = Kuu_inv.matmul(self.polynomial_coef.unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)
		w_var = Cff.sub(Kuu_inv.matmul(Cuf.unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)).mul(self.kernel.outputscale)
		return w_mean, w_var

	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Union[Tensor, None]]:
		"""
		x_mean & x_var ~ [B, Q, C]
		->
		f_mean & f_var ~ [B, D, K]
		"""
		w_mean, w_var = self.compute_w(x_mean, x_var)
		f_mean = w_mean.sum(-2)
		f_var = None if w_var is None else w_var.sum(-2) + 1e-6
		return f_mean, f_var

	def extra_repr(self) -> str:
		return f"in_dim={self.in_dim}, out_dim={self.out_dim}, num_induc={self.num_induc}, num_comp={self.num_comp}"


class NormLayer(Layer):

	def __init__(self):
		Layer.__init__(self)

	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Union[Tensor, None]]:
		return x_mean.tanh(), None if x_var is None else x_var.tanh()


