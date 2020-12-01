import torch

from torch.distributions import Categorical
from functools import reduce
from typing import Tuple

from torch.distributions import OneHotCategorical

import torch.nn as nn

class MultiCategorical(Categorical):
    """
        customized distribution to deal with multiple categorical data
        Example::
        >>> m = MultiCategorical(torch.tensor([[0.3, 0.2, 0.4, 0.1, 0.25, 0.5, 0.25, 0.3, 0.4, 0.1, 0.1, 0.1],
                                        [0.2, 0.3, 0.1, 0.4, 0.5, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1]]), (4, 3, 5))
        >>> m.sample()
        tensor([[0, 1, 4],
                [1, 0, 3]])
    """

    def __init__(self, probs: torch.Tensor, sections: Tuple):
        self._sections = sections
        self._dists = [Categorical(x) for x in torch.split(probs, sections, dim=-1)]

    def sample(self, sample_shape=torch.Size()):
        """
        concat sample from each one-hot custom together
        :param sample_shape:
        :return: [sample_dist1, sample_dist2, ...]
        """
        res = torch.cat([dist.sample().unsqueeze(-1) for dist in self._dists], dim=-1)
        return res

    def log_prob(self, value):
        values = torch.split(value, 1, dim=-1)
        log_probs = [dist.log_prob(v.squeeze()) for dist, v in zip(self._dists, values)]
        return reduce(torch.add, log_probs)

    def entropy(self):
        entropy_list = [dist.entropy() for dist in self._dists]
        return reduce(torch.add, entropy_list)



class MultiOneHotCategorical(OneHotCategorical):
    """
        customized distribution to deal with multiple one-hot categorical data
        Example::
        >>> m = MultiOneHotCategorical(torch.tensor([[0.3, 0.2, 0.4, 0.1, 0.25, 0.5, 0.25, 0.3, 0.4, 0.1, 0.1, 0.1],
                                        [0.2, 0.3, 0.1, 0.4, 0.5, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1]]), (4, 3, 5))
        >>> m.sample()
        tensor([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
                [0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.]])
    """

    def __init__(self, probs: torch.Tensor, sections: Tuple):
        self._sections = sections
        self._dists = [OneHotCategorical(x) for x in torch.split(probs, sections, dim=-1)]

    def sample(self, sample_shape=torch.Size()):
        """
        concat sample from each one-hot custom together
        :param sample_shape:
        :return: [sample_dist1, sample_dist2, ...]
        """
        res = torch.cat([dist.sample() for dist in self._dists], dim=-1)
        return res

    def log_prob(self, value):
        values = torch.split(value, self._sections, dim=-1)
        log_probs = [dist.log_prob(v) for dist, v in zip(self._dists, values)]
        return reduce(torch.add, log_probs)

    def entropy(self):
        entropy_list = [dist.entropy() for dist in self._dists]
        return reduce(torch.add, entropy_list)


class MultiSoftMax(nn.Module):
    r"""customized module to deal with multiple softmax case.
    softmax feature: [dim_begin, dim_end)
    sections define sizes of each softmax case
    Examples::
        >>> m = MultiSoftMax(dim_begin=0, dim_end=5, sections=(2, 3))
        >>> input = torch.randn((2, 5))
        >>> output = m(input)
    """

    def __init__(self, dim_begin: int, dim_end: int, sections: Tuple = None):
        super().__init__()
        self.dim_begin = dim_begin
        self.dim_end = dim_end
        self.sections = sections

        if sections:
            assert dim_end - dim_begin == sum(sections), "expected same length of sections and customized" \
                                                         "dims"

    def forward(self, input_tensor: torch.Tensor):
        x = input_tensor[..., self.dim_begin:self.dim_end]
        res = input_tensor.clone()
        res[..., self.dim_begin:self.dim_end] = torch.cat([
            xx.softmax(dim=-1) for xx in torch.split(x, self.sections, dim=-1)], dim=-1)
        return res