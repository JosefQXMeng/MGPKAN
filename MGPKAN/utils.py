
from typing import Optional

import torch
from torch import Tensor



def SquaredExponentialExpectation(
	l_sq: Tensor, mu: Optional[Tensor] = None, sigma_sq: Optional[Tensor] = None,
) -> Tensor:
	var_sum = l_sq if sigma_sq is None else l_sq.add(sigma_sq)
	std_quot = torch.ones([]) if sigma_sq is None else l_sq.div(var_sum).sqrt()
	exp = torch.ones([]) if mu is None else mu.pow(2).div(var_sum).div(-2).exp()
	return std_quot.mul(exp)


