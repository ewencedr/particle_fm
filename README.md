<div align="center">

# EPiC Flow Matching

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.9+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the official repository implementing the EPiC Flow Matching point cloud generative machine learning models from arxiv1111.11111.

EPiC Flow Matching is a [Continuous Normalising Flow](https://arxiv.org/abs/1806.07366) that is trained with a simulation free approach called [Flow Matching](https://arxiv.org/abs/2210.02747). The model uses [DeepSet](https://arxiv.org/abs/1703.06114) based [EPiC layers](https://arxiv.org/abs/2301.08128) for the architecture, which allow for good scalability to high set sizes.

Additionally to the EPiC Flow Matching model, this repository also implements various other loss functions that correspond to other flow matching/diffusion based models, like [Conditional Flow Matching](https://arxiv.org/abs/2302.00482) and [DDIM](https://arxiv.org/abs/2010.02502) based [PC-Jedi](https://arxiv.org/abs/2303.05376).

The models are tested on the [JetNet dataset](https://zenodo.org/record/6975118). The JetNet dataset is used in particle physics to test point cloud generative deep learning architectures. It consists of simulated particle jets produced by proton proton collisions in a simplified detector. The dataset is split into jets originating from tops, light quarks, gluons, W bosons and Z bosons and has a maximum number of 150 particles per jet.

This repository uses [pytorch lightning](https://www.pytorchlightning.ai/index.html), [hydra](https://hydra.cc/docs/intro/) for model configurations and supports logging with [comet](https://www.comet.com/site/) and [wandb](https://wandb.ai/site). For a deeper explanation of how to use this repository, please have a look at the [template](https://github.com/ashleve/lightning-hydra-template) directly.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Create .env file to set paths and API keys

```bash
PROJEKT_ROOT="/folder/folder/"
DATA_DIR="/folder/folder/"
LOG_DIR="/folder/folder/"
COMET_API_TOKEN="XXXXXXXXXX"
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

The experiments include

<details>
  <summary>
    <b>fm_tops30_cond</b>
  </summary>
  EPiC Flow Matching trained on top30 dataset with conditioning on jet mass and pt
</details>
<details>
  <summary>
    <b>fm_tops30</b>
  </summary>
  EPiC Flow Matching trained on top30 dataset with no additional conditioning. Jet size conditioning is a neccessity for the architecture
</details>
<details>
  <summary>
    <b>fm_tops150_cond</b>
  </summary>
  EPiC Flow Matching trained on top150 dataset with conditioning on jet mass and pt
</details>
<details>
  <summary>
    <b>fm_tops150</b>
  </summary>
  EPiC Flow Matching trained on top150 dataset with no additional conditioning. Jet size conditioning is a neccessity for the architecture
</details>
<details>
  <summary>
    <b>fm_alljet150_cond</b>
  </summary>
  EPiC Flow Matching trained on all jet types with a maximum of 150 particles per jet and conditioning on jet mass and pt.
</details>
<details>
  <summary>
    <b>diffusion_tops30_cond</b>
  </summary>
  EPiC Jedi (DDIM diffusion based) trained on top30 dataset with conditioning on jet mass and pt
</details>
<details>
  <summary>
    <b>diffusion_tops30</b>
  </summary>
  EPiC Jedi (DDIM diffusion based) trained on top30 dataset with no additional conditioning. Jet size conditioning is a neccessity for the architecture
</details>
<details>
  <summary>
    <b>diffusion_tops150_cond</b>
  </summary>
  EPiC Jedi (DDIM diffusion based) trained on top150 dataset with conditioning on jet mass and pt
</details>
<details>
  <summary>
    <b>diffusion_tops150</b>
  </summary>
  EPiC Jedi (DDIM diffusion based) trained on top150 dataset with no additional conditioning. Jet size conditioning is a neccessity for the architecture
</details>
<details>
  <summary>
    <b>diffusion_alljet150_cond</b>
  </summary>
  EPiC Jedi (DDIM diffusion based) trained on all jet types with a maximum of 150 particles per jet and conditioning on jet mass and pt.
</details>
<br>

During training and evaluation, metrics and plots can be logged via comet and wandb. After training the model will be evaluated automatically and the final results will be saved locally and logged via the selected loggers. The evaluation can also be manually started like this

```bash
python src/eval.py experiment=experiment_name.yaml ckpt_path=checkpoint_path
```

Notebooks are available to quickly train, evaluate models and create plots.
