import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm


class EPiC_layer(nn.Module):
    """
    equivariant layer with global concat & residual connections inside this module  & weight_norm
    ordered: first update global, then local
    """

    def __init__(self, local_in_dim, hid_dim, latent_dim):
        super().__init__()
        self.fc_global1 = weight_norm(nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim))
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))

    def forward(self, x_global, x_local):  # shapes: x_global[b,latent], x_local[b,n,latent_local]
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        # meansum pooling
        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooledCATglobal = torch.cat(
            [x_pooled_mean, x_pooled_sum, x_global], 1
        )  # meansum pooling

        # phi global
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))  # new intermediate step
        x_global = F.leaky_relu(
            self.fc_global2(x_global1) + x_global
        )  # with residual connection before AF

        x_global2local = x_global.view(-1, 1, latent_global).repeat(
            1, n_points, 1
        )  # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)

        # phi p
        x_local1 = F.leaky_relu(
            self.fc_local1(x_localCATglobal)
        )  # with residual connection before AF
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


# Decoder / Generator for multiple particles with Variable Number of Equivariant Layers (with global concat)
# added same global and local usage in EPiC layer
# order: global first, then local
args = {"latent"}


class EPiC_generator(nn.Module):
    def __init__(
        self,
        latent=16,
        latent_local=3,
        hid_d=256,
        feats=128,
        equiv_layers=8,
        return_latent_space=False,
    ):
        super().__init__()
        self.latent = latent  # used for latent size of equiv concat
        self.latent_local = latent_local  # noise
        self.hid_d = hid_d  # default 256
        self.feats = feats
        self.equiv_layers = equiv_layers
        self.return_latent_space = return_latent_space  # false or true

        self.local_0 = weight_norm(nn.Linear(self.latent_local, self.hid_d))
        self.global_0 = weight_norm(nn.Linear(self.latent, self.hid_d))
        self.global_1 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(EPiC_layer(self.hid_d, self.hid_d, self.latent))

        self.local_1 = weight_norm(nn.Linear(self.hid_d, self.feats))

    def forward(self, z_global, z_local):  # shape: [batch, points, feats]

        batch_size, _, _ = z_local.size()
        latent_tensor = z_global.clone().reshape(batch_size, 1, -1)

        z_local = F.leaky_relu(self.local_0(z_local))
        z_global = F.leaky_relu(self.global_0(z_global))
        z_global = F.leaky_relu(self.global_1(z_global))

        latent_tensor = torch.cat([latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1)

        z_global_in, z_local_in = z_global.clone(), z_local.clone()

        # equivariant connections, each one_hot conditined
        for i in range(self.equiv_layers):
            z_global, z_local = self.nn_list[i](z_global, z_local)  # contains residual connection

            z_global, z_local = (
                z_global + z_global_in,
                z_local + z_local_in,
            )  # skip connection to sampled input

            latent_tensor = torch.cat(
                [latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1
            )
        # final local NN to get down to input feats size
        out = self.local_1(z_local)
        if self.return_latent_space:
            return out, latent_tensor
        else:
            return out  # [batch, points, feats]
