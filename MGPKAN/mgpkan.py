
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleDict

from .abstr import Network, Regr
from .layers import FCLayer, NormLayer



class MGPKAN(Network):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, list[int]]],
		num_induc: Union[int, list[int]],
		num_comp: Union[int, list[int]] = 1,
		kernel : list[Optional[str]] = "SquaredExponentialKernel",
	):
		Network.__init__(self, in_dim, out_dim, hidden_dims)

		if isinstance(num_induc, int):
			num_induc = [num_induc] * (len(self.dims)-1)
		assert len(num_induc) == len(self.dims)-1

		if isinstance(num_comp, int):
			num_comp = [num_comp] * (len(self.dims)-1)
		assert len(num_comp) == len(self.dims)-1

		if kernel is None or isinstance(kernel, str):
			kernel = [kernel] * (len(self.dims)-1)
		assert len(kernel) == len(self.dims)-1

		layerdict = {}
		for i in range(len(self.dims)-1):
			if i > 0:
				layerdict[f"norm_{i}"] = NormLayer()
			layerdict[f"fc_{i+1}"] = FCLayer(
				self.dims[i], self.dims[i+1], num_induc[i], num_comp[i], kernel[i],
			)
		self.layers = ModuleDict(layerdict)
		self.inputs_normalized = False

	def forward(self, x: Tensor) -> tuple[Tensor, Union[Tensor, None]]:
		"""
		x ~ [B, D^0]
		->
		f_mean & f_var ~ [K^L, B, D^L]
		"""
		f_mean = x.unsqueeze(-1)
		f_var = None
		for layer in self.layers.values():
			f_mean, f_var = layer.forward(f_mean, f_var)
		f_mean = f_mean.movedim(-1,0)
		f_var = f_var.movedim(-1,0)
		return f_mean, f_var

	def normalize_inputs(self) -> None:
		if not self.inputs_normalized:
			layers = ModuleDict({"norm_0": NormLayer()})
			layers.update(self.layers)
			self.layers = layers
			self.inputs_normalized = True
	
	def extra_repr(self) -> str:
		return "shape=[{}]".format(",".join(map(str, self.dims)))


class MGPKANR(MGPKAN, Regr):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, list[int]]],
		num_induc: Union[int, list[int]],
		num_comp: Union[int, list[int]] = 1,
		kernel : list[Optional[str]] = "SquaredExponentialKernel",
		min_noise_var: float = 1e-6,
	):
		MGPKAN.__init__(self, in_dim, out_dim, hidden_dims, num_induc, num_comp, kernel)
		Regr.__init__(self, out_dim, min_noise_var)

	def loglikelihood(self, x: Tensor, y: Tensor) -> Tensor:
		f_mean, f_var = self.forward(x)
		return self.ell(f_mean, f_var, y).sum(-1).mean()
	
	def pred(self, x: Tensor, y: Optional[Tensor] = None) -> Any:
		with torch.no_grad():
			f_mean, f_var = self.forward(x)
			pred_mean = f_mean.mean(0)
			if y is None:
				mll = None
			else:
				mll = self.mll(f_mean, f_var, y)
				alpha = mll.max(0).values
				mll = mll.sub(alpha).exp().mean(0).log().add(alpha)
			return pred_mean, mll


