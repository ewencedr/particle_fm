import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.pylogger import get_pylogger

logger_el = get_pylogger("epic_layer")
logger_eg = get_pylogger("epic_generator")
logger_ed = get_pylogger("epic_discriminator")
logger_emask = get_pylogger("epic_mask")


class EPiC_layer(nn.Module):
    """equivariant layer with global concat & residual connections inside this module  & weight_norm
    ordered: first update global, then local

    Args:
        local_in_dim (int, optional): local in dim. Defaults to 3.
        hid_dim (int, optional): hidden dimension. Defaults to 256.
        latent_dim (int, optional): latent dim. Defaults to 16.
        global_cond_dim (int, optional): Global conditioning dimension. 0 corresponds to no conditioning. Defaults to 0.
        local_cond_dim (int, optional): Local conditioning dimension. 0 corresponds to no conditioning. Defaults to 0.
        activation (str, optional): Activation function to use in architecture. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper for linear layers. Defaults to "weight_norm".
        frequencies (int, optional): Frequencies for time. Basically half the size of the time vector that is added to the model. Defaults to 6.
        num_points (int, optional): Number of points in set. Defaults to 30.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        sum_scale (float, optional): Factor that is multiplied with the sum pooling. Defaults to 1e-2.
    """

    def __init__(
        self,
        local_in_dim: int = 3,
        hid_dim: int = 256,
        latent_dim: int = 16,
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        frequencies: int = 6,
        num_points: int = 30,
        dropout: float = 0.0,
        sum_scale: float = 1e-2,
    ):
        super().__init__()
        self.activation = activation
        self.global_cond_dim = global_cond_dim
        self.local_cond_dim = local_cond_dim

        self.num_points = num_points
        self.sum_scale = sum_scale

        self.t_local_cat = t_local_cat
        self.t_global_cat = t_global_cat
        t_local_dim = 2 * frequencies if self.t_local_cat else 0
        t_global_dim = 2 * frequencies if self.t_global_cat else 0

        self.wrapper_func = getattr(nn.utils, wrapper_func, lambda x: x)
        self.fc_global1 = self.wrapper_func(
            nn.Linear(
                int(2 * hid_dim) + latent_dim + t_global_dim + self.global_cond_dim,
                hid_dim,
            )
        )
        self.fc_global2 = self.wrapper_func(
            nn.Linear(hid_dim + t_global_dim + self.global_cond_dim, latent_dim)
        )
        self.fc_local1 = self.wrapper_func(
            nn.Linear(local_in_dim + latent_dim + t_local_dim + self.local_cond_dim, hid_dim)
        )
        self.fc_local2 = self.wrapper_func(
            nn.Linear(hid_dim + t_local_dim + self.local_cond_dim, hid_dim)
        )

        self.do = nn.Dropout(dropout)

    def forward(
        self,
        t: torch.Tensor = None,
        x_global: torch.Tensor = None,
        x_local: torch.Tensor = None,
        global_cond_in: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shapes: x_global[b,latent], x_local[b,n,input_dim]
        # Check and prepare input
        if x_global is None or x_local is None:
            raise ValueError("x_global or x_local is None")

        if global_cond_in is None and (self.global_cond_dim > 0 or self.local_cond_dim > 0):
            raise ValueError(
                f"global_cond_dim is {self.global_cond_dim} and local_cond_dim is {self.local_cond_dim} but no global_cond is given"
            )

        if t is None and (self.t_local_cat or self.t_global_cat):
            raise ValueError(
                f"t_local_cat is {self.t_local_cat} and t_global_cat is {self.t_global_cat} but no t is given"
            )

        # conditioning

        if global_cond_in is None:
            global_cond = torch.Tensor().to(x_global.device)

        if self.global_cond_dim == 0:
            global_cond = torch.Tensor().to(x_global.device)
        else:
            global_cond = global_cond_in

        if self.local_cond_dim == 0:
            local_cond = torch.Tensor().to(x_local.device)
        else:
            local_cond = global_cond_in.unsqueeze(-2).repeat_interleave(self.num_points, dim=-2)
            logger_el.debug(f"local_cond shape: {local_cond.shape}")

        if global_cond_in is not None:
            logger_el.debug(f"global_cond_in shape: {global_cond_in.shape}")
            logger_el.debug(f"global_cond_in repeat shape: {global_cond_in.shape}")

        # time conditioning
        if t is None:
            t = torch.Tensor().to(x_global.device)
        else:
            logger_el.debug(f"t shape: {t.shape}")

        if not self.t_local_cat:
            t_local = torch.Tensor().to(t.device)
        else:
            t_local = t

        if self.t_global_cat:
            # prepare t for concat to global
            logger_el.debug(f"t shape: {t.shape}")
            logger_el.debug(f"t: {t[:3]}")
            t_global = t.clone()[:, 0, :]
            logger_el.debug(f"t_global shape: {t_global.shape}")
        else:
            t_global = torch.Tensor().to(t.device)

        # mask

        if mask is None:
            logger_emask.debug("mask is None")
            mask = torch.ones_like(x_local[:, :, 0]).unsqueeze(-1)

        # actual forward pass

        batch_size, n_points, input_dim = x_local.size()
        latent_global = x_global.size(1)
        logger_emask.debug(f"mask.shape: {mask.shape}")

        # meansum pooling
        x_pooled_sum = (x_local * mask).sum(1, keepdim=False)
        x_pooled_mean = x_pooled_sum / mask.sum(1, keepdim=False)
        x_pooled_sum = x_pooled_sum * self.sum_scale
        x_pooledCATglobal = torch.cat(
            [
                x_pooled_mean,
                x_pooled_sum,
                x_global,
            ],
            1,
        )  # meansum pooling
        logger_el.debug(f"x_pooled_mean.shape: {x_pooled_mean.shape}")
        logger_el.debug(f"x_pooled_sum.shape: {x_pooled_sum.shape}")
        logger_el.debug(f"x_global.shape: {x_global.shape}")

        # phi global
        logger_el.debug(f"t.shape: {t.shape}")
        logger_el.debug(f"x_pooledCATglobal.shape: {x_pooledCATglobal.shape}")

        x_global1 = getattr(F, self.activation, lambda x: x)(
            self.fc_global1(torch.cat((t_global, x_pooledCATglobal, global_cond), dim=-1))
        )  # new intermediate step
        logger_el.debug(f"x_global1.shape: {x_global1.shape}")
        x_global = getattr(F, self.activation, lambda x: x)(
            self.fc_global2(torch.cat((t_global, x_global1, global_cond), dim=-1)) + x_global
        )  # with residual connection before AF
        x_global = self.do(x_global)

        x_global2local = x_global.view(-1, 1, latent_global).repeat(
            1, n_points, 1
        )  # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)

        # phi p
        logger_el.debug(f"x_localCATglobal.shape: {x_localCATglobal.shape}")
        x_local1 = getattr(F, self.activation, lambda x: x)(
            self.fc_local1(torch.cat((t_local, x_localCATglobal, local_cond), dim=-1))
        )  # with residual connection before AF
        logger_el.debug(f"x_local1.shape: {x_local1.shape}")
        x_local = getattr(F, self.activation, lambda x: x)(
            self.fc_local2(torch.cat((t_local, x_local1, local_cond), dim=-1)) + x_local
        )
        x_local = self.do(x_local)

        return x_global, x_local


class EPiC_encoder(nn.Module):
    """Encoder based on EPiC Discriminator.

    Args:
        latent (int, optional): used for latent size of equiv concat. Defaults to 16.
        input_dim (int, optional): number of features of input point cloud. Defaults to 3.
        hid_d (int, optional): Hidden dimension. Defaults to 256.
        feats (int, optional): Embedding dimension for EPiC Layers. Defaults to 128.
        equiv_layers (int, optional): Number of EPiC Layers used. Defaults to 8.
        global_cond_dim (int, optional): Global conditioning dimension. 0 corresponds to no conditioning. Defaults to 0.
        local_cond_dim (int, optional): Local conditioning dimension. 0 corresponds to no conditioning. Defaults to 0.
        activation (str, optional): Activation function to use in architecture. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper for linear layers. Defaults to "weight_norm".
        frequencies (int, optional): Frequencies for time. Basically half the size of the time vector that is added to the model. Defaults to 6.
        num_points (int, optional): Number of points in set. Defaults to 30.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector in EPiC layers. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        sum_scale (float, optional): Scale for sum pooling. Defaults to 1e-2.
    """

    def __init__(
        self,
        latent: int = 16,
        input_dim: int = 3,
        hid_d: int = 256,
        feats: int = 128,
        equiv_layers: int = 8,
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        frequencies: int = 6,
        num_points: int = 30,
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        dropout: float = 0.0,
        sum_scale: float = 1e-2,
    ):
        super().__init__()
        self.activation = activation
        self.latent = latent
        self.input_dim = input_dim
        self.hid_d = hid_d
        self.feats = feats
        self.equiv_layers = equiv_layers
        self.global_cond_dim = global_cond_dim
        self.local_cond_dim = local_cond_dim
        self.num_points = num_points
        self.sum_scale = sum_scale

        self.t_local_cat = t_local_cat
        self.t_global_cat = t_global_cat
        t_local_dim = 2 * frequencies if self.t_local_cat else 0
        t_global_dim = 2 * frequencies if self.t_global_cat else 0

        self.wrapper_func = getattr(nn.utils, wrapper_func, lambda x: x)

        self.fc_l1 = self.wrapper_func(
            nn.Linear(self.input_dim + t_local_dim + self.local_cond_dim, self.hid_d)
        )
        self.fc_l2 = self.wrapper_func(
            nn.Linear(self.hid_d + t_local_dim + self.local_cond_dim, self.hid_d)
        )

        self.fc_g1 = self.wrapper_func(
            nn.Linear(int(2 * self.hid_d) + t_global_dim + self.global_cond_dim, self.hid_d)
        )
        self.fc_g2 = self.wrapper_func(
            nn.Linear(self.hid_d + t_global_dim + self.global_cond_dim, self.latent)
        )

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(
                EPiC_layer(
                    self.hid_d,
                    self.hid_d,
                    self.latent,
                    activation=activation,
                    wrapper_func=wrapper_func,
                    num_points=self.num_points,
                    t_global_cat=t_global_cat,
                    t_local_cat=t_local_cat,
                    global_cond_dim=global_cond_dim,
                    local_cond_dim=local_cond_dim,
                    frequencies=frequencies,
                    dropout=dropout,
                    sum_scale=sum_scale,
                )
            )

        self.fc_l3 = self.wrapper_func(
            nn.Linear(self.hid_d + t_local_dim + self.local_cond_dim, self.feats),
        )

        self.do = nn.Dropout(dropout)

    def forward(
        self,
        t_in: torch.Tensor = None,
        x_local: torch.Tensor = None,
        global_cond_in: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):  # shape: [batch, points, feats]
        if x_local is None:
            raise ValueError("x_local is None")
        if global_cond_in is None and (self.global_cond_dim > 0 or self.local_cond_dim > 0):
            raise ValueError(
                f"global_cond_dim is {self.global_cond_dim} and local_cond_dim is {self.local_cond_dim} but no global_cond is given"
            )
        if t_in is None and (self.t_local_cat or self.t_global_cat):
            raise ValueError(
                f"t_local_cat is {self.t_local_cat} and t_global_cat is {self.t_global_cat} but no t is given"
            )
        if t_in is None:
            t = torch.Tensor().to(x_local.device)
        else:
            t = t_in

        if mask is None:
            mask = torch.ones_like(x_local[:, :, 0]).unsqueeze(-1)

        logger_eg.debug(f"t: {t.shape}")
        logger_eg.debug(f"x_local: {x_local.shape}")

        # time conditioning
        if not self.t_local_cat:
            t_local = torch.Tensor().to(t.device)
        else:
            t_local = t

        if self.t_global_cat:
            # prepare t for concat to global
            t_global = t.clone()[:, 0, :]
        else:
            t_global = torch.Tensor().to(t.device)

        # global conditioning
        if self.global_cond_dim == 0:
            global_cond = torch.Tensor().to(x_local.device)
        else:
            global_cond = global_cond_in

        # local conditioning
        if self.local_cond_dim > 0:
            local_cond = global_cond_in.unsqueeze(-2).repeat_interleave(self.num_points, dim=-2)
            logger_eg.debug(f"local_cond shape: {local_cond.shape}")
        else:
            local_cond = torch.Tensor().to(x_local.device)

        # actual forward pass
        x_local = getattr(F, self.activation, lambda x: x)(
            self.fc_l1(torch.cat((t_local, x_local, local_cond), dim=-1))
        )

        x_local = getattr(F, self.activation, lambda x: x)(
            self.fc_l2(torch.cat((t_local, x_local, local_cond), dim=-1)) + x_local
        )
        x_local = self.do(x_local)

        z_sum = (x_local * mask).sum(1, keepdim=False)
        z_mean = z_sum / mask.sum(1, keepdim=False)
        z_sum = z_sum * self.sum_scale

        x_global = torch.cat((z_sum, z_mean), dim=-1)

        x_global = getattr(F, self.activation, lambda x: x)(
            self.fc_g1(torch.cat((t_global, x_global, global_cond), dim=-1))
        )
        x_global = getattr(F, self.activation, lambda x: x)(
            self.fc_g2(torch.cat((t_global, x_global, global_cond), dim=-1))
        )

        for i in range(self.equiv_layers):
            x_global, x_local = self.nn_list[i](
                t_in, x_global, x_local, global_cond_in=global_cond_in, mask=mask
            )

        x_local = getattr(F, self.activation, lambda x: x)(
            self.fc_l3(torch.cat((t_local, x_local, local_cond), dim=-1))
        )

        return x_local * mask
