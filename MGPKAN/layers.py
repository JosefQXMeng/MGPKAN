
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
		learnable_weight: bool = False,
	):
		Layer.__init__(self, in_dim, out_dim)
		
		self.num_induc = num_induc  # M
		self.num_comp = num_comp  # K

		self.kernel = SquaredExponentialKernel([out_dim, in_dim, num_comp])
		# z ~ [D, Q, K, M]
		self.induc_loc = Parameter(torch.rand(out_dim, in_dim, num_comp, num_induc).mul(2).sub(1))
		# u ~ [D, Q, K, M]
		self.induc_value = Parameter(torch.zeros(out_dim, in_dim, num_comp, num_induc))
		# w ~ [(K)]
		self._mix_weight = Parameter(torch.zeros(num_comp-1)) if learnable_weight else None

	@property
	def mix_weight(self) -> Union[Tensor, None]:
		return None if self._mix_weight is None else torch.cat([self._mix_weight, torch.zeros([1])]).softmax(0)

	def compute_w(
		self, x_mean: Tensor, x_var: Optional[Tensor] = None, weight: Optional[Tensor] = None,
	) -> tuple[Tensor, Tensor]:
		"""
		x_mean & x_var ~ [B, Q, C]
		->
		w_mean & w_var ~ [B, D, Q, K]
		"""

		self.kernel.transform_lengthscale()
		# Kuu_inv ~ [D, Q, K, M, M]
		Kuu_inv = self.kernel.Kuu_inv(self.induc_loc)
		# Cuf ~ [B, D, Q, K, M]
		Cuf = self.kernel.Cuf(self.induc_loc, x_mean, x_var, weight)
		# Cff ~ [(B, D, Q, K)]
		Cff = self.kernel.Cff(x_mean, x_var, weight)

		w_mean = Kuu_inv.matmul(self.induc_value.unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)
		w_var = Cff.sub(Kuu_inv.matmul(Cuf.unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)).mul(self.kernel.outputscale)
		return w_mean, w_var

	def forward(
		self, x_mean: Tensor, x_var: Optional[Tensor] = None, weight: Optional[Tensor] = None,
	) -> tuple[Tensor, Tensor, Union[Tensor, None]]:
		"""
		x_mean & x_var ~ [B, Q, C]
		->
		f_mean & f_var ~ [B, D, K]
		"""
		w_mean, w_var = self.compute_w(x_mean, x_var, weight)
		f_mean = w_mean.sum(-2)
		f_var = w_var.sum(-2) + 1e-6
		return f_mean, f_var, self.mix_weight

	def extra_repr(self) -> str:
		return f"in_dim={self.in_dim}, out_dim={self.out_dim}\nnum_induc={self.num_induc}, num_comp={self.num_comp}"


class NormLayer(Layer):

	def __init__(self):
		Layer.__init__(self)

	def forward(
		self, x_mean: Tensor, x_var: Optional[Tensor] = None, weight: Optional[Tensor] = None,
	) -> tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
		return x_mean.tanh(), None if x_var is None else x_var.tanh(), weight


