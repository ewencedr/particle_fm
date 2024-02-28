from torch import nn
import torch
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden,
        dropout,
        weightnorm=True,
        glu=False,
        critic=True,
        t_local_cat=False,
        t_global_cat=False,
        global_cond_dim=0,
        frequencies=8,
        local_cat_cond=False,
        global_cat_cond=False,
    ):
        super().__init__()

        self.fc0 = nn.Linear(hidden + 2 * frequencies * t_local_cat + local_cat_cond, hidden)
        self.fc0_cls = nn.Linear(
            embed_dim + 2 * frequencies * t_global_cat + global_cat_cond, hidden
        )
        self.fc1 = nn.Linear(hidden + embed_dim + local_cat_cond, hidden)
        self.glu = False
        self.fc1_cls = nn.Linear(
            hidden + 1 + global_cond_dim + 2 * frequencies * t_global_cat, embed_dim
        )
        self.fc2_cls = nn.Linear(
            embed_dim + 2 * frequencies * t_global_cat + global_cat_cond, embed_dim
        )
        self.cond_cls = nn.Linear(global_cond_dim, hidden)  # embed_dim if cond_dim == 1 else
        # embed_dim if cond_dim == 1 else
        self.attn = nn.MultiheadAttention(
            hidden,
            num_heads,
            batch_first=True,
        )

        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)
        self.t_local_cat = t_local_cat
        self.t_global_cat = t_global_cat
        self.local_cat_cond = local_cat_cond
        self.global_cat_cond = global_cat_cond

    def forward(self, x, x_cls, cond, mask, t_in=None):
        res = x.clone()

        # res_cls = x_cls.clone()
        if self.t_local_cat:
            x = torch.cat((x, t_in), dim=-1)
        if self.t_global_cat:
            x_cls = torch.cat((x_cls, t_in[:, :1, :]), dim=-1)
        if self.global_cat_cond:
            x_cls = torch.cat((x_cls, cond[..., -1:]), dim=-1)
        if self.local_cat_cond:
            x = torch.cat((x, cond[..., -1:].expand(-1, x.shape[1], 1)), dim=-1)
        x = self.fc0(self.act(x))

        x_cls = self.ln(self.fc0_cls(self.act(x_cls)))
        x_cls, w = self.attn(
            x_cls, x, x, key_padding_mask=~mask.bool().squeeze(-1), need_weights=False
        )
        x_cls = (
            torch.cat((x_cls, cond), dim=-1)
            if not self.t_global_cat
            else torch.cat((x_cls, cond, t_in[:, :1, :]), dim=-1)
        )
        x_cls = self.fc1_cls(x_cls)
        if self.glu:
            x_cls = self.act(F.glu(torch.cat((x_cls, self.cond_cls(cond)), dim=-1)))
        x_cls = x_cls if not self.t_global_cat else torch.cat((x_cls, t_in[:, :1, :]), dim=-1)
        x_cls = x_cls if not self.global_cat_cond else torch.cat((x_cls, cond[..., -1:]), dim=-1)
        x_cls = self.fc2_cls(x_cls)
        if self.local_cat_cond:
            x = torch.cat((x, cond[..., -1:].expand(-1, x.shape[1], 1)), dim=-1)
        x = self.fc1(torch.cat((x, x_cls.expand(-1, x.shape[1], -1)), dim=-1)) + res
        return x, x_cls, w


class MDMA(nn.Module):
    def __init__(
        self,
        latent: int = 16,
        input_dim: int = 3,
        hidden_dim: int = 256,
        feats: int = 128,
        layers: int = 16,
        global_cond_dim: int = 0,
        local_cond_dim: int = 0,
        activation: str = "leaky_relu",
        wrapper_func: str = "",
        frequencies: int = 6,
        num_points: int = 30,
        t_local_cat: bool = True,
        t_global_cat: bool = True,
        dropout: float = 0.0,
        sum_scale: float = 1e-2,
        avg_n: int = 30,
        num_heads: int = 8,
        local_cat_cond: bool = False,
        global_cat_cond: bool = False,
        **kwargs
    ):
        self.t_local_cat = t_local_cat
        self.t_global_cat = t_global_cat

        super().__init__()
        self.embbed = nn.Linear(
            input_dim + 2 * frequencies * t_local_cat + local_cat_cond, hidden_dim
        )
        self.embbed_cls = nn.Linear(hidden_dim + 1 + global_cond_dim, latent)
        self.encoder = nn.ModuleList(
            [
                Block(
                    embed_dim=latent,
                    num_heads=num_heads,
                    hidden=hidden_dim,
                    weightnorm=False,
                    dropout=0,
                    glu=False,
                    critic=False,
                    t_local_cat=t_local_cat,
                    t_global_cat=t_global_cat,
                    global_cond_dim=global_cond_dim,
                    frequencies=frequencies,
                    local_cat_cond=local_cat_cond,
                    global_cat_cond=global_cat_cond,
                )
                for i in range(layers)
            ]
        )
        self.out = nn.Linear(hidden_dim + local_cat_cond, 1)
        self.act = nn.LeakyReLU()
        self.avg_n = avg_n
        self.local_cat_cond = local_cat_cond
        self.global_cat_cond = global_cat_cond
        self.cond = nn.Linear(global_cond_dim + 1, latent)
        self.global_cond = global_cond_dim > 0

    def forward(
        self,
        t_in: torch.Tensor = None,
        x: torch.Tensor = None,
        global_cond_in: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):

        if self.t_local_cat:
            x = torch.cat((x, t_in), dim=-1)
        if self.local_cat_cond:
            x = torch.cat((x, global_cond_in.unsqueeze(-1).expand(-1, x.shape[1], 1)), dim=-1)
        x = self.act(self.embbed(x))
        x[~mask.bool().squeeze(-1)] = 0
        x_cls = x.sum(1).unsqueeze(1).clone() / self.avg_n
        x_cls = torch.cat((x_cls, mask.sum(1, keepdim=True).reshape(-1, 1, 1)), dim=-1)

        if self.global_cat_cond or self.global_cond:
            x_cls = torch.cat((x_cls, global_cond_in.unsqueeze(-1)), dim=-1)
        x_cls = self.embbed_cls(x_cls)
        cond = mask.sum(1, keepdim=True).reshape(-1, 1, 1)
        if self.global_cond or self.global_cat_cond:
            cond = torch.cat((cond, global_cond_in.unsqueeze(-1)), dim=-1)
        x_cls = F.glu(torch.cat((x_cls, self.cond(cond.float())), dim=-1))
        for layer in self.encoder:
            x, x_cls, w = layer(x, x_cls=x_cls, mask=mask.bool(), cond=cond, t_in=t_in)
        if self.local_cat_cond:
            x = torch.cat((x, global_cond_in.unsqueeze(-1).expand(-1, x.shape[1], 1)), dim=-1)
        x = self.out(self.act(x))
        return x * mask
