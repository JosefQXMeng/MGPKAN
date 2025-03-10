
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleDict

from .abstr import Network, Regr
from .layers import FCLayer, NormLayer


class GPKAN(Network):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, list[int]]],
		num_induc: Union[int, list[int]],
		num_comp: Union[int, list[int]] = 1,
		weighted: Union[bool, list[bool]] = False,
		degree: Union[int, list[int]] = 0,
	):
		Network.__init__(self, in_dim, out_dim, hidden_dims)

		if isinstance(num_induc, int):
			num_induc = [num_induc] * (len(self.dims)-1)
		assert len(num_induc) == len(self.dims)-1

		if isinstance(num_comp, int):
			num_comp = [num_comp] * (len(self.dims)-1)
		assert len(num_comp) == len(self.dims)-1

		if isinstance(weighted, bool):
			weighted = [weighted] * (len(self.dims)-1)
		assert len(weighted) == len(self.dims)-1

		if isinstance(degree, int):
			degree = [degree] * (len(self.dims)-1)
		assert len(degree) == len(self.dims)-1

		layerdict = {}
		for i in range(len(self.dims)-1):
			layerdict[f"norm_{i}"] = NormLayer()
			layerdict[f"fc_{i+1}"] = FCLayer(
				self.dims[i], self.dims[i+1], num_induc[i], num_comp[i], weighted[i], degree[i],
			)
		self.layers = ModuleDict(layerdict)

	def forward(self, x: Tensor) -> tuple[Tensor, Union[Tensor, None]]:
		"""
		x ~ [B, D^0]
		->
		f_mean & f_var ~ [B, D^L]
		"""
		f_mean = x
		f_var = None
		for layer in self.layers.values():
			f_mean, f_var = layer.forward(f_mean, f_var)
		return f_mean, f_var


class GPKANR(GPKAN, Regr):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, list[int]]],
		num_induc: Union[int, list[int]],
		num_comp: Union[int, list[int]] = 1,
		weighted: Union[bool, list[bool]] = False,
		degree: Union[int, list[int]] = 0,
		min_noise_var: float = 1e-6,
	):
		GPKAN.__init__(self, in_dim, out_dim, hidden_dims, num_induc, num_comp, weighted, degree)
		Regr.__init__(self, out_dim, min_noise_var)

	def loglikelihood(self, x: Tensor, y: Tensor) -> Tensor:
		f_mean, f_var = self.forward(x)
		return self.ell(f_mean, f_var, y).sum(-1).mean()
	
	def pred(self, x: Tensor, y: Optional[Tensor] = None) -> tuple[Tensor, Union[Tensor, None]]:
		with torch.no_grad():
			f_mean, f_var = self.forward(x)
			mll = None if y is None else self.mll(f_mean, f_var, y)
			return f_mean, mll


