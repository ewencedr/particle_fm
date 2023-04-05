import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.pylogger import get_pylogger

logger_el = get_pylogger("epic_layer")
logger_eg = get_pylogger("epic_generator")


class EPiC_layer(nn.Module):
    """equivariant layer with global concat & residual connections inside this module  & weight_norm
    ordered: first update global, then local

    Args:
        local_in_dim (int, optional): local in dim. Defaults to 3.
        hid_dim (int, optional): hidden dimension. Defaults to 256.
        latent_dim (int, optional): latent dim. Defaults to 16.
        global_cond_dim (int, optional): Global conditioning dimension. 0 corresponds to no conditioning. Defaults to 0.
        activation (str, optional): Activation function to use in architecture. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper for linear layers. Defaults to "weight_norm".
        frequencies (int, optional): Frequencies for time. Defaults to 6.
        num_points (int, optional): Number of points in set. Defaults to 30.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
    """

    def __init__(
        self,
        local_in_dim: int = 3,
        hid_dim: int = 256,
        latent_dim: int = 16,
        global_cond_dim: int = 0,
        t_local_cat: bool = False,
        t_global_cat: bool = False,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        frequencies: int = 6,
        num_points: int = 30,
    ):

        super().__init__()
        self.activation = activation
        self.global_cond_dim = global_cond_dim

        self.t_local_cat = t_local_cat
        self.t_global_cat = t_global_cat
        t_local_dim = 2 * frequencies if self.t_local_cat else 0
        t_global_dim = latent_dim if self.t_global_cat else 0

        self.wrapper_func = getattr(nn.utils, wrapper_func, lambda x: x)

        self.fc_global1 = self.wrapper_func(
            nn.Linear(int(2 * hid_dim) + latent_dim + t_global_dim + global_cond_dim, hid_dim)
        )
        self.fc_global2 = self.wrapper_func(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = self.wrapper_func(
            nn.Linear(local_in_dim + latent_dim + t_local_dim, hid_dim)
        )
        self.fc_local2 = self.wrapper_func(nn.Linear(hid_dim + t_local_dim, hid_dim))
        if self.t_global_cat:
            self.fc_t = self.wrapper_func(nn.Linear(2 * frequencies * num_points, latent_dim))

    def forward(
        self,
        t: torch.Tensor,
        x_global: torch.Tensor,
        x_local: torch.Tensor,
        global_cond: torch.Tensor = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # shapes: x_global[b,latent], x_local[b,n,latent_local]
        if global_cond is None and self.global_cond_dim > 0:
            raise ValueError(
                f"global_cond_dim is {self.global_cond_dim} but no global_cond is given"
            )
        if global_cond is None:
            global_cond = torch.Tensor().to(x_global.device)

        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)
        logger_el.debug(f"t shape: {t.shape}")

        if not self.t_local_cat:
            t = torch.Tensor().to(t.device)
        if self.t_global_cat:
            # prepare t for concat to global
            t_global = self.fc_t(t.clone().reshape(t.shape[0], -1))
            logger_el.debug(f"t_global shape: {t_global.shape}")
        else:
            t_global = torch.Tensor().to(t.device)

        # meansum pooling
        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooledCATglobal = torch.cat(
            [x_pooled_mean, x_pooled_sum, x_global, t_global, global_cond],
            1,
        )  # meansum pooling
        logger_el.debug(f"x_pooled_mean.shape: {x_pooled_mean.shape}")
        logger_el.debug(f"x_pooled_sum.shape: {x_pooled_sum.shape}")
        logger_el.debug(f"x_global.shape: {x_global.shape}")

        # phi global
        logger_el.debug(f"t.shape: {t.shape}")
        logger_el.debug(f"x_pooledCATglobal.shape: {x_pooledCATglobal.shape}")
        x_global1 = getattr(F, self.activation, lambda x: x)(
            self.fc_global1(x_pooledCATglobal)
        )  # new intermediate step
        logger_el.debug(f"x_global1.shape: {x_global1.shape}")
        x_global = getattr(F, self.activation, lambda x: x)(
            self.fc_global2(x_global1) + x_global
        )  # with residual connection before AF

        x_global2local = x_global.view(-1, 1, latent_global).repeat(
            1, n_points, 1
        )  # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)

        # phi p
        logger_el.debug(f"x_localCATglobal.shape: {x_localCATglobal.shape}")
        x_local1 = getattr(F, self.activation, lambda x: x)(
            self.fc_local1(torch.cat((t, x_localCATglobal), dim=-1))
        )  # with residual connection before AF
        logger_el.debug(f"x_local1.shape: {x_local1.shape}")
        x_local = getattr(F, self.activation, lambda x: x)(
            self.fc_local2(torch.cat((t, x_local1), dim=-1)) + x_local
        )

        return x_global, x_local


class EPiC_generator(nn.Module):
    """Decoder / Generator for multiple particles with Variable Number of Equivariant Layers (with global concat)
       added same global and local usage in EPiC layer
       order: global first, then local


    Args:
        latent (int, optional): used for latent size of equiv concat. Defaults to 16.
        latent_local (int, optional): noise. Defaults to 3.
        hid_d (int, optional): Hidden dimension. Defaults to 256.
        feats (int, optional): Embedding dimension for EPiC Layers. Defaults to 128.
        equiv_layers (int, optional): Number of EPiC Layers used. Defaults to 8.
        global_cond_dim (int, optional): Global conditioning dimension. 0 corresponds to no conditioning. Defaults to 0.
        return_latent_space (bool, optional): Return latent space. Defaults to False.
        activation (str, optional): Activation function to use in architecture. Defaults to "leaky_relu".
        wrapper_func (str, optional): Wrapper for linear layers. Defaults to "weight_norm".
        frequencies (int, optional): Frequencies for time. Defaults to 6.
        num_points (int, optional): Number of points in set. Defaults to 30.
        t_local_cat (bool, optional): Concat time to local linear layers. Defaults to False.
        t_global_cat (bool, optional): Concat time to global vector. Defaults to False.
    """

    def __init__(
        self,
        latent: int = 16,
        latent_local: int = 3,
        hid_d: int = 256,
        feats: int = 128,
        equiv_layers: int = 8,
        global_cond_dim: int = 0,
        return_latent_space: bool = False,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        frequencies: int = 6,
        num_points: int = 30,
        t_local_cat: bool = False,
        t_global_cat: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.latent = latent
        self.latent_local = latent_local
        self.hid_d = hid_d
        self.feats = feats
        self.equiv_layers = equiv_layers
        self.global_cond_dim = global_cond_dim
        self.return_latent_space = return_latent_space

        self.t_local_cat = t_local_cat
        self.t_global_cat = t_global_cat
        t_local_dim = 2 * frequencies if self.t_local_cat else 0
        t_global_dim = self.latent if self.t_global_cat else 0

        self.wrapper_func = getattr(nn.utils, wrapper_func, lambda x: x)
        self.local_0 = self.wrapper_func(nn.Linear(self.latent_local + t_local_dim, self.hid_d))
        self.global_0 = self.wrapper_func(nn.Linear(self.latent + t_global_dim, self.hid_d))
        self.global_1 = self.wrapper_func(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(
                EPiC_layer(
                    self.hid_d,
                    self.hid_d,
                    self.latent,
                    activation=activation,
                    wrapper_func=wrapper_func,
                    num_points=num_points,
                    t_global_cat=t_global_cat,
                    t_local_cat=t_local_cat,
                    global_cond_dim=global_cond_dim,
                )
            )

        self.local_1 = self.wrapper_func(nn.Linear(self.hid_d + t_local_dim, self.feats))
        if self.t_global_cat:
            self.fc_t = self.wrapper_func(nn.Linear(2 * frequencies * num_points, self.latent))

    def forward(
        self,
        t: torch.Tensor,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        global_cond: torch.Tensor = None,
    ):  # shape: [batch, points, feats]
        batch_size, _, _ = z_local.size()
        latent_tensor = z_global.clone().reshape(batch_size, 1, -1)
        logger_eg.debug(f"t: {t.shape}")
        logger_eg.debug(f"z_local: {z_local.shape}")

        if not self.t_local_cat:
            t = torch.Tensor().to(t.device)
        if self.t_global_cat:
            # prepare t for concat to global
            t_global = getattr(F, self.activation, lambda x: x)(
                self.fc_t(t.clone().reshape(t.shape[0], -1))
            )
            logger_el.debug(f"t_global shape: {t_global.shape}")
            z_global = torch.cat([z_global, t_global], 1)

        z_local = getattr(F, self.activation, lambda x: x)(
            self.local_0(torch.cat((t, z_local), dim=-1))
        )
        logger_eg.debug(f"z_global1: {z_global.shape}")
        z_global = getattr(F, self.activation, lambda x: x)(self.global_0(z_global))
        logger_eg.debug(f"z_global2: {z_global.shape}")
        z_global = getattr(F, self.activation, lambda x: x)(self.global_1(z_global))

        latent_tensor = torch.cat([latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1)

        z_global_in, z_local_in = z_global.clone(), z_local.clone()
        # equivariant connections, each one_hot conditined
        for i in range(self.equiv_layers):
            z_global, z_local = self.nn_list[i](
                t, z_global, z_local, global_cond=global_cond
            )  # contains residual connection

            z_global, z_local = (
                z_global + z_global_in,
                z_local + z_local_in,
            )  # skip connection to sampled input

            latent_tensor = torch.cat(
                [latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1
            )
        # final local NN to get down to input feats size
        logger_eg.debug(f"z_local2: {z_local.shape}")
        out = self.local_1(torch.cat((t, z_local), dim=-1))
        if self.return_latent_space:
            return out, latent_tensor
        else:
            return out  # [batch, points, feats]
