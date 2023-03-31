import torch
import torch.nn as nn
import torch.nn.functional as F


class EPiC_layer(nn.Module):
    """
    equivariant layer with global concat & residual connections inside this module  & weight_norm
    ordered: first update global, then local
    """

    def __init__(
        self,
        local_in_dim,
        hid_dim,
        latent_dim,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        frequencies=6,
    ):
        super().__init__()
        self.activation = activation
        self.wrapper_func = getattr(nn.utils, wrapper_func, lambda x: x)

        self.fc_global1 = self.wrapper_func(nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim))
        self.fc_global2 = self.wrapper_func(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = self.wrapper_func(
            nn.Linear(local_in_dim + latent_dim + 2 * frequencies, hid_dim)
        )
        self.fc_local2 = self.wrapper_func(nn.Linear(hid_dim + 2 * frequencies, hid_dim))

    def forward(
        self, t, x_global, x_local
    ):  # shapes: x_global[b,latent], x_local[b,n,latent_local]
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        # meansum pooling
        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooledCATglobal = torch.cat(
            [x_pooled_mean, x_pooled_sum, x_global], 1
        )  # meansum pooling

        # phi global
        # print(f"t.shape epic: {t.shape}")
        # print(f"x_pooledCATglobal.shape epic: {x_pooledCATglobal.shape}")
        x_global1 = getattr(F, self.activation, lambda x: x)(
            self.fc_global1(x_pooledCATglobal)
        )  # new intermediate step
        # print(f"x_global1.shape epic: {x_global1.shape}")
        x_global = getattr(F, self.activation, lambda x: x)(
            self.fc_global2(x_global1) + x_global
        )  # with residual connection before AF

        x_global2local = x_global.view(-1, 1, latent_global).repeat(
            1, n_points, 1
        )  # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)

        # phi p
        # print(f"x_localCATglobal.shape epic: {x_localCATglobal.shape}")
        x_local1 = getattr(F, self.activation, lambda x: x)(
            self.fc_local1(torch.cat((t, x_localCATglobal), dim=-1))
        )  # with residual connection before AF
        # print(f"x_local1.shape epic: {x_local1.shape}")
        x_local = getattr(F, self.activation, lambda x: x)(
            self.fc_local2(torch.cat((t, x_local1), dim=-1)) + x_local
        )

        return x_global, x_local


# Decoder / Generator for multiple particles with Variable Number of Equivariant Layers (with global concat)
# added same global and local usage in EPiC layer
# order: global first, then local
class EPiC_generator(nn.Module):
    def __init__(
        self,
        latent=16,
        latent_local=3,
        hid_d=256,
        feats=128,
        equiv_layers=8,
        return_latent_space=False,
        activation: str = "leaky_relu",
        wrapper_func: str = "weight_norm",
        frequencies=6,
    ):
        super().__init__()
        self.activation = activation
        self.latent = latent  # used for latent size of equiv concat
        self.latent_local = latent_local  # noise
        self.hid_d = hid_d  # default 256
        self.feats = feats
        self.equiv_layers = equiv_layers
        self.return_latent_space = return_latent_space  # false or true
        self.wrapper_func = getattr(nn.utils, wrapper_func, lambda x: x)
        self.local_0 = self.wrapper_func(
            nn.Linear(self.latent_local + 2 * frequencies, self.hid_d)
        )
        self.global_0 = self.wrapper_func(nn.Linear(self.latent, self.hid_d))
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
                )
            )

        self.local_1 = self.wrapper_func(nn.Linear(self.hid_d + 2 * frequencies, self.feats))

    def forward(self, t, z_global, z_local):  # shape: [batch, points, feats]

        batch_size, _, _ = z_local.size()
        latent_tensor = z_global.clone().reshape(batch_size, 1, -1)
        # print(f"t: {t.shape}")
        # print(f"z_local: {z_local.shape}")
        z_local = getattr(F, self.activation, lambda x: x)(
            self.local_0(torch.cat((t, z_local), dim=-1))
        )
        # print(f"z_global1: {z_global.shape}")
        z_global = getattr(F, self.activation, lambda x: x)(self.global_0(z_global))
        # print(f"z_global2: {z_global.shape}")
        z_global = getattr(F, self.activation, lambda x: x)(self.global_1(z_global))

        latent_tensor = torch.cat([latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1)

        z_global_in, z_local_in = z_global.clone(), z_local.clone()

        # equivariant connections, each one_hot conditined
        for i in range(self.equiv_layers):
            z_global, z_local = self.nn_list[i](
                t, z_global, z_local
            )  # contains residual connection

            z_global, z_local = (
                z_global + z_global_in,
                z_local + z_local_in,
            )  # skip connection to sampled input

            latent_tensor = torch.cat(
                [latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1
            )
        # final local NN to get down to input feats size
        # print(f"z_local2: {z_local.shape}")
        out = self.local_1(torch.cat((t, z_local), dim=-1))
        if self.return_latent_space:
            return out, latent_tensor
        else:
            return out  # [batch, points, feats]
