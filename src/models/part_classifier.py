import lightning as L
import torch
import torch.nn as nn
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.train import optim
from weaver.utils.nn.tools import _flatten_label, _flatten_preds

# standard model configuration from
# https://github.com/jet-universe/particle_transformer/blob/main/networks/example_ParticleTransformer.py#L26-L44  # noqa: E501
part_default_kwargs = dict(
    input_dim=7,
    num_classes=10,
    # network configurations
    pair_input_dim=4,
    use_pre_activation_pair=False,
    embed_dims=[128, 512, 128],
    pair_embed_dims=[64, 64, 64],
    num_heads=8,
    num_layers=8,
    num_cls_layers=2,
    block_params=None,
    cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
    fc_params=[],
    activation="gelu",
    # misc
    trim=True,
    for_inference=False,
)


class ParticleTransformerPL(L.LightningModule):
    """Pytorch-lightning wrapper for ParticleTransformer."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)
        self.loss_func = torch.nn.CrossEntropyLoss()
        # self.data_config = data_config
        self.fc_params = kwargs["fc_params"]
        self.num_classes = kwargs["num_classes"]
        self.last_embed_dim = kwargs["embed_dims"][-1]
        self.set_learning_rates()
        self.test_output = []
        self.test_labels = []

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)

    def training_step(self, batch, batch_nb):
        # X, y, _ = batch
        part_features, part_coords, part_mask, part_labels = batch
        # TODO: lorentz vectors should come from dataloader
        part_lorentz = torch.randn(part_features.size(0), 4, part_features.size(2)).to("cuda")
        # inputs = [X[k].to("cuda") for k in self.data_config.input_names]
        # label = y[self.data_config.label_names[0]].long()
        label = part_labels.long().to("cuda")
        model_output = self(
            points=None,
            features=part_features.to("cuda"),
            lorentz_vectors=part_lorentz.to("cuda"),
            mask=part_mask.to("cuda"),
        )
        with torch.cuda.amp.autocast():
            logits = _flatten_preds(model_output)
            loss = self.loss_func(logits, label)
        _, preds = logits.max(1)
        # correct = (preds == label).sum().item()
        # accuracy = correct / label.size(0)
        # self.log("train_loss", loss, on_step=True, on_epoch=True)
        # self.log("train_acc", accuracy, on_step=True, on_epoch=True)
        return loss

    # def training_step(self, batch, batch_nb):
    #     X, y, _ = batch
    #     inputs = [X[k].to("cuda") for k in self.data_config.input_names]
    #     label = y[self.data_config.label_names[0]].long()
    #     label = label.to("cuda")
    #     try:
    #         label_mask = y[self.data_config.label_names[0] + "_mask"].bool()
    #     except KeyError:
    #         label_mask = None
    #     label = _flatten_label(label, label_mask)
    #     model_output = self(*inputs)
    #     with torch.cuda.amp.autocast():
    #         logits = _flatten_preds(model_output)
    #         loss = self.loss_func(logits, label)
    #     _, preds = logits.max(1)
    #     correct = (preds == label).sum().item()
    #     accuracy = correct / label.size(0)
    #     self.log("train_loss", loss, on_step=True, on_epoch=True)
    #     self.log("train_acc", accuracy, on_step=True, on_epoch=True)
    #     return loss

    # def validation_step(self, batch, batch_nb):
    #     X, y, _ = batch
    #     inputs = [X[k].to("cuda") for k in self.data_config.input_names]
    #     label = y[self.data_config.label_names[0]].long()
    #     label = label.to("cuda")
    #     try:
    #         label_mask = y[self.data_config.label_names[0] + "_mask"].bool()
    #     except KeyError:
    #         label_mask = None
    #     label = _flatten_label(label, label_mask)
    #     model_output = self(*inputs)
    #     with torch.cuda.amp.autocast():
    #         logits = _flatten_preds(model_output)
    #         loss = self.loss_func(logits, label)
    #     _, preds = logits.max(1)
    #     correct = (preds == label).sum().item()
    #     accuracy = correct / label.size(0)
    #     self.log("val_loss", loss, on_step=True, on_epoch=True)
    #     self.log("val_acc", accuracy, on_step=True, on_epoch=True)
    #     return loss

    # def test_step(self, batch, batch_nb):
    #     X, y, _ = batch
    #     inputs = [X[k].to("cuda") for k in self.data_config.input_names]
    #     label = y[self.data_config.label_names[0]].long()
    #     label = label.to("cuda")
    #     self.mod.for_inference = True
    #     self.eval()
    #     model_output = self(*inputs)
    #     self.test_output.append(model_output.detach().cpu().numpy())
    #     self.test_labels.append(label.detach().cpu().numpy())

    def set_learning_rates(self, lr_fc=0.001, lr=0.0001):
        """Set the learning rates for the fc layer and the rest of the model."""
        self.lr_fc = lr_fc
        self.lr = lr
        print(f"Setting learning rates to lr_fc={self.lr_fc} and lr={self.lr}")

    def reinitialise_fc(self):
        """Reinitialise the final fully connected network of the model."""
        if self.fc_params is not None:
            fcs = []
            in_dim = self.last_embed_dim
            for out_dim, drop_rate in self.fc_params:
                fcs.append(
                    nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate))
                )
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, self.num_classes))
            self.mod.fc = nn.Sequential(*fcs)
        else:
            self.mod.fc = None

    def set_args_for_optimizer(self, args, dev):
        self.opt_args = args
        self.dev = dev

    def configure_optimizers(self):
        # optimizer, scheduler = optim(self.opt_args, self, self.dev)
        # User adamw optimizer
        # TODO: use more sophisticated optimizer and scheduler?
        # compare to the standard ParT fine-tuning
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
