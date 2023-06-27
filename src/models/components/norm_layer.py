from typing import Optional, Union

import torch as T
import torch.nn as nn


class IterativeNormLayer(nn.Module):
    """A basic normalisation layer so it can be part of the model.

    Note! If a mask is provided in the forward pass, then this must be the dimension to apply over
    the masked inputs! For example: Graph nodes are usually batch x n_nodes x features so to
    normalise over the features one would typically give extra_dims as (0,) But nodes are always
    passed with the mask which flattens it to batch x features. Batch dimension is done
    automatically, so we dont pass any extra_dims!!!
    """

    def __init__(
        self,
        inpt_dim: Union[T.Tensor, tuple, int],
        means: Optional[T.Tensor] = None,
        vars: Optional[T.Tensor] = None,
        n: int = 0,
        max_n: int = 5_00_000,
        extra_dims: Union[tuple, int] = (),
    ) -> None:
        """Init method for Normalisatiion module.

        Args:
            inpt_dim: Shape of the input tensor, required for reloading
            means: Calculated means for the mapping. Defaults to None.
            vars: Calculated variances for the mapping. Defaults to None.
            n: Number of samples used to make the mapping. Defaults to None.
            max_n: Maximum number of iterations before the means and vars are frozen
            extra_dims: The extra dimension(s) over which to calculate the stats
                Will always calculate over the batch dimension
        """
        super().__init__()

        # Fail if only one of means or vars is provided
        if (means is None) ^ (vars is None):  # XOR
            raise ValueError(
                """Only one of 'means' and 'vars' is defined. Either both or
                neither must be defined"""
            )

        # Allow integer inpt_dim and n arguments
        if isinstance(inpt_dim, int):
            inpt_dim = (inpt_dim,)
        if isinstance(n, int):
            n = T.tensor(n)

        # The dimensions over which to apply the normalisation, make positive!
        if isinstance(extra_dims, int):  # Ensure it is a list
            extra_dims = [extra_dims]
        else:
            extra_dims = list(extra_dims)
        if any([abs(e) > len(inpt_dim) for e in extra_dims]):  # Check size
            raise ValueError("extra_dims argument lists dimensions outside input range")
        for d in range(len(extra_dims)):
            if extra_dims[d] < 0:  # make positive
                extra_dims[d] = len(inpt_dim) + extra_dims[d]
            extra_dims[d] += 1  # Add one because we are inserting a batch dimension
        self.extra_dims = extra_dims

        # Calculate the input and output shapes
        self.max_n = max_n
        self.inpt_dim = list(inpt_dim)
        self.stat_dim = [1] + list(inpt_dim)  # Add batch dimension
        for d in range(len(self.stat_dim)):
            if d in self.extra_dims:
                self.stat_dim[d] = 1

        # Buffers are needed for saving/loading the layer
        self.register_buffer("means", T.zeros(self.stat_dim) if means is None else means)
        self.register_buffer("vars", T.ones(self.stat_dim) if vars is None else vars)
        self.register_buffer("n", n)

        # For the welford algorithm it is useful to have another variable m2
        self.register_buffer("m2", T.ones(self.stat_dim) if vars is None else vars)

        # If the means are set here then the model is "frozen" and not updated
        self.frozen = means is not None

    def _mask(self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None) -> T.Tensor:
        if mask is None:
            return inpt
        return inpt[mask]

    def _check_attributes(self) -> None:
        if self.means is None or self.vars is None:
            raise ValueError("Stats for have not been initialised or fit() has not been run!")

    def fit(
        self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None, freeze: bool = True
    ) -> None:
        """Set the stats given a population of data."""
        inpt = self._mask(inpt, mask)
        print(f"fit input shape: {inpt.shape}")
        self.vars, self.means = T.var_mean(inpt, dim=(0, *self.extra_dims), keepdim=True)
        print(f"fit means shape: {self.means.shape}")
        self.n = T.tensor(len(inpt), device=self.means.device)
        self.m2 = self.vars * self.n
        self.frozen = freeze

    def forward(self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None) -> T.Tensor:
        """Applies the standardisation to a batch of inputs, also uses the inputs to update the
        running stats if in training mode."""
        with T.no_grad():
            print(f"forward input shape: {inpt.shape}")
            sel_inpt = self._mask(inpt, mask)
            print(f"forward sel_inpt shape: {sel_inpt.shape}")
            if not self.frozen and self.training:
                self.update(sel_inpt)

            # Apply the mapping
            print(f"forward sel_input2 shape: {sel_inpt.shape}")
            print(f"forward means shape: {self.means.shape}")
            normed_inpt = (sel_inpt - self.means) / (self.vars.sqrt() + 1e-8)
            print(f"forward means shape: {self.means.shape}")
            print(f"forward normed_inpt shape: {normed_inpt.shape}")

            # Undo the masking
            if mask is not None:
                inpt = inpt.clone()  # prevents inplace operation, bad for autograd
                inpt[mask] = normed_inpt
                return inpt

            return normed_inpt

    def reverse(self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None) -> T.Tensor:
        """Unnormalises the inputs given the recorded stats."""
        sel_inpt = self._mask(inpt, mask)
        unnormed_inpt = sel_inpt * self.vars.sqrt() + self.means

        # Undo the masking
        if mask is not None:
            inpt = inpt.clone()  # prevents inplace operation, bad for autograd
            inpt[mask] = unnormed_inpt
            return inpt

        return unnormed_inpt

    def update(self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None) -> None:
        """Update the running stats using a batch of data."""
        print(f"update input shape: {inpt.shape}")
        inpt = self._mask(inpt, mask)
        print(f"update input shape2: {inpt.shape}")
        # For first iteration
        if self.n == 0:
            self.fit(inpt, freeze=False)
            return

        # later iterations based on batched welford algorithm
        with T.no_grad():
            self.n += len(inpt)
            delta = inpt - self.means
            self.means += (delta / self.n).mean(dim=(0, *self.extra_dims), keepdim=True) * len(
                inpt
            )
            delta2 = inpt - self.means
            self.m2 += (delta * delta2).mean(dim=(0, *self.extra_dims), keepdim=True) * len(inpt)
            self.vars = self.m2 / self.n

        # Freeze the model if we exceed the requested stats
        self.frozen = self.n >= self.max_n
