from typing import Any, List

import torch
from torch import nn
from torchdyn.core import NeuralODE
from torchdyn.models import CNF, autograd_trace, hutch_trace
from torchdyn.nn import DataControl, DepthCat, Augmenter
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch.distributions import (
    MultivariateNormal,
    Uniform,
    TransformedDistribution,
    SigmoidTransform,
    Categorical,
)


class CNFLitModule(LightningModule):
    """Continous Normalizing Flow Module for training generative models.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        trace: callable = autograd_trace,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.StepLR,
    ) -> None:
        """Initialize the model.
        Args:
            f: torch module
            optimizer: torch optimizer
            datamodule: LightningDataModule needs to have "dim" property
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # stochastic estimators require a definition of a distribution where "noise" vectors are sampled from
        self.noise_dist = MultivariateNormal(torch.zeros(2), torch.eye(2))

        self.net = net
        self.iters = 0
        # cnf wraps the net as with other energy models
        self.cnf = CNF(self.net, trace_estimator=trace)
        self.nde = NeuralODE(
            self.cnf, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )

        # Augmenter takes care of setting up the additional scalar dimension for the divergence dynamics. The DEFunc wrapper (implicitly defined when passing f to the NeuralDE) will ensure compatibility of depth-concatenation and data-control with the divergence dimension.
        # Utilizing additional augmented dimensions is also compatible, as only the first will be used for the jacobian trace.
        self.model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), self.nde)

        self.register_buffer("prior_mean", torch.zeros(2).to(self.device))
        self.register_buffer("prior_cov", torch.eye(2).to(self.device))
        self.prior = MultivariateNormal(self.prior_mean, self.prior_cov)

        # loss function

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        pass

    def training_step(self, batch: Any, batch_idx: int):
        self.iters += 1
        x, _ = batch
        t, xtrJ = self.model(x)
        # print(f"xtrJ device: {xtrJ.device}")
        # print(f"self.prior device: {self.prior}")
        self.prior = MultivariateNormal(self.prior_mean, self.prior_cov)
        logprob = (
            self.prior.log_prob(xtrJ[1, :, 1:]) - xtrJ[1, :, 0]
        )  # logp(z_S) = logp(z_0) - \int_0^S trJ
        loss = -torch.mean(logprob)
        self.nde.nfe = 0
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        t, xtrJ = self.model(x)
        self.prior = MultivariateNormal(self.prior_mean, self.prior_cov)
        logprob = (
            self.prior.log_prob(xtrJ[1, :, 1:]) - xtrJ[1, :, 0]
        )  # logp(z_S) = logp(z_0) - \int_0^S trJ
        loss = -torch.mean(logprob)
        self.nde.nfe = 0
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def sample(self, num_samples):
        sample = self.prior.sample(torch.Size([num_samples]))
        # integrating from 1 to 0
        self.model[1].t_span = torch.linspace(1, 0, 2)
        new_x = self.model(sample)[1]
        return new_x


if __name__ == "__main__":
    _ = CNFLitModule(None, None, None, None, None)
