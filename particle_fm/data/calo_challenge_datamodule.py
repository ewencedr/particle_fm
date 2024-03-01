import copy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import rv_continuous
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    TensorDataset,
    random_split,
)

from .components import (
    DQ,
    DQLinear,
    LogitTransformer,
    ScalerBase,
    ScalerBaseNew,
    SqrtTransformer,
)


class CustomDataset(Dataset):
    def __init__(self, data, E):
        assert len(data) == len(E), "The lengths of data and E are not equal"
        self.data = data
        self.E = E

    def __getitem__(self, index):
        return self.data[index], self.E[index]

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Sort sequences by length
        indices = sorted(indices, key=lambda x: len(self.data_source[x]))
        # Create batches based on the sorted indices
        batches = [
            indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            np.random.shuffle(batches)
        if self.drop_last or len(batches[-1]) == 0:
            batches = batches[:-1]
        yield from batches

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def pad_collate_fn(batch):

    batch, E = zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)[:, :, :4].float()
    mask = ~(
        torch.arange(max_len).expand(len(batch), max_len)
        < torch.tensor([len(sample) for sample in batch]).unsqueeze(1)
    )
    E = (torch.stack(E).log() - 10).float()
    return padded_batch, ~mask.unsqueeze(-1), E


def pad_collate_aug_fn(batch, scaler=None):

    batch, E = zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)[:, :, :4].float()
    mask = ~(
        torch.arange(max_len).expand(len(batch), max_len)
        < torch.tensor([len(sample) for sample in batch]).unsqueeze(1)
    )
    E = (torch.stack(E).log() - 10).float()
    if scaler is not None:
        org_shape = padded_batch.shape
        temp = padded_batch[~mask]
        temp = scaler.inverse_transform(temp.double())
        # diff=temp[...,1:]-temp[...,1:].floor()
        temp[:, 2] += torch.randint(0, 16, (temp.shape[0],), device=padded_batch.device).double()
        temp[:, 2] %= 16
        padded_batch[~mask] = scaler.transform(temp.double()).float()
        # padded_batch[...,1:]+=diff.float()
        padded_batch = padded_batch.reshape(*org_shape)

    return padded_batch, mask, E


def pad_collate_aug_fn_big(batch, scaler=None):

    batch, E = zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)[:, :, :4].float()
    mask = ~(
        torch.arange(max_len).expand(len(batch), max_len)
        < torch.tensor([len(sample) for sample in batch]).unsqueeze(1)
    )
    E = (torch.stack(E).log() - 10).float()
    if scaler is not None:
        org_shape = padded_batch.shape
        temp = padded_batch[~mask]
        temp = scaler.inverse_transform(temp.double())
        # diff=temp[...,1:]-temp[...,1:].floor()
        temp[:, 2] += torch.randint(0, 50, (temp.shape[0],), device=padded_batch.device).double()
        temp[:, 2] %= 50
        padded_batch[~mask] = scaler.transform(temp.double()).float()
        # padded_batch[...,1:]+=diff.float()
        padded_batch = padded_batch.reshape(*org_shape)

    return padded_batch, mask, E


# Pad the sequences using pad_sequence()


class BucketBatchSamplerMax(BatchSampler):
    def __init__(
        self, data_source, batch_size, max_tokens_per_batch=400000, shuffle=True, drop_last=False
    ):
        self.data_source = data_source
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Sort sequences by length
        indices = sorted(indices, key=lambda x: len(self.data_source[x]))
        # Create batches based on the total number of tokens per batch
        batches = []
        batch = []
        batch_tokens = 0
        for idx in indices:
            sample_len = len(self.data_source[idx])
            if (
                batch_tokens + sample_len > self.max_tokens_per_batch
                or len(batch) >= self.batch_size
            ):

                batches.append(batch)
                batch = []
                batch_tokens = 0
            batch.append(idx)
            batch_tokens += sample_len
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class CaloChallengeDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors."""

    def __init__(self, name, batch_size, augmentation, max=False, **kwargs):
        self.name = name
        self.batch_size = batch_size
        self.max = max
        self.new = True if name.find("new") > -1 else False
        self.cart = True if name.find("cart") > -1 else False
        self.aug = augmentation
        self.save_hyperparameters(logger=False)
        self.variable_jet_sizes = False
        super().__init__()

    def setup(self, stage):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        self.data = torch.load(f"/beegfs/desy/user/kaechben/calochallenge/pc_train_{self.name}.pt")
        self.E = self.data["energies"]
        self.data = self.data["data"]
        self.val_data = torch.load(
            f"/beegfs/desy/user/kaechben/calochallenge/pc_train_{self.name}.pt"
        )

        # self.val_data=torch.load(f"/beegfs/desy/user/kaechben/calochallenge/pc_test_{self.name}.pt")
        self.val_E = self.val_data["energies"]
        self.val_data = self.val_data["data"]

        self.scaler = ScalerBaseNew(
            transfs=[], featurenames=["E", "z", "alpha", "r"], name=self.name, overwrite=False
        )
        del self.scaler.transfs[1].steps[0]
        if self.max:
            self.train_iterator = BucketBatchSamplerMax(
                self.data,
                batch_size=self.batch_size,
                drop_last=True,
                max_tokens_per_batch=400000,
                shuffle=True,
            )
            self.val_iterator = BucketBatchSamplerMax(
                self.val_data,
                batch_size=self.batch_size,
                max_tokens_per_batch=400000,
                drop_last=False,
                shuffle=True,
            )
        else:
            self.train_iterator = BucketBatchSampler(
                self.data, batch_size=self.batch_size, drop_last=True, shuffle=True
            )
            self.val_iterator = BucketBatchSampler(
                self.val_data, batch_size=self.batch_size, drop_last=False, shuffle=True
            )
        if self.aug:
            self.train_dl = DataLoader(
                CustomDataset(self.data, self.E),
                batch_sampler=self.train_iterator,
                collate_fn=lambda x: (
                    pad_collate_aug_fn(x, self.scaler)
                    if not self.name.find("big") != -1
                    else pad_collate_aug_fn_big(x, self.scaler)
                ),
                num_workers=16,
            )
        else:
            self.train_dl = DataLoader(
                CustomDataset(self.data, self.E),
                batch_sampler=self.train_iterator,
                collate_fn=pad_collate_fn,
                num_workers=16,
            )
        self.val_dl = DataLoader(
            CustomDataset(self.val_data, self.val_E),
            batch_sampler=self.val_iterator,
            collate_fn=pad_collate_fn,
            num_workers=16,
        )

    def train_dataloader(self):
        return (
            self.train_dl
        )  # DataLoader(self.data, batch_size=10, shuffle=False, num_workers=1, drop_last=False,collate_fn=point_cloud_collate_fn)

    def val_dataloader(self):
        return self.val_dl


if __name__ == "__main__":

    loader = CaloChallengeDataloader("middle_new_linear", 64, max=False, augmentation=False)
    loader.setup("train")

    # for i in loader.val_dataloader():
    #     if i[0].shape[1]>0:
    #         print(i[0][:,:,0].min())

    #         assert (i[0]==i[0]).all()
    mins = torch.ones(4).unsqueeze(0)
    maxs = torch.ones(4).unsqueeze(0)

    for i in loader.train_dataloader():
        if len(i[0]) > 0:
            i[0][i[1].squeeze(-1)] = loader.scaler.inverse_transform(
                i[0][i[1].squeeze(-1)].double()
            )
            mins = torch.min(
                torch.cat((mins, i[0][i[1].squeeze(-1)].min(0, keepdim=True)[0]), dim=0), dim=0
            )[0].unsqueeze(0)
            maxs = torch.max(
                torch.cat((maxs, i[0][i[1].squeeze(-1)].max(0, keepdim=True)[0]), dim=0), dim=0
            )[0].unsqueeze(0)
