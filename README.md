<div align="center">

# Particle-FM/Diffusion

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

[![arxiv](http://img.shields.io/badge/arXiv-2310.00049-B31B1B.svg)](https://arxiv.org/abs/2310.00049)
[![arxiv](http://img.shields.io/badge/arXiv-2310.06897-B31B1B.svg)](https://arxiv.org/abs/2310.06897)
[![Paper](http://img.shields.io/badge/arxiv-2312.00123-B31B1B.svg)](https://arxiv.org/abs/2312.00123)

</div>

## Description

This repository contains multiple (mostly) generative neural networks and multiple datasets from particle physics. The focus lies on continuous time generative models for point clouds. Diffusion/ Score-based models / Continuous Normalizing Flows are in this repository combined under the Flow Matching framework, where these models differ only in the loss function.

## 🤖 Models

### Generative Models

#### Architectures:

- [EPiC](https://arxiv.org/abs/2301.08128) (for Sets, based on [DeepSets](https://arxiv.org/abs/1703.06114))
- [Full Transformer](https://arxiv.org/abs/2307.06836) (for Sets)
- [CrossAttention Transformer](https://arxiv.org/abs/2307.06836) (for Sets)
- Fully Connected

#### Loss Functions:

- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Conditional Flow Matching](https://arxiv.org/abs/2302.00482)
- [OT Conditional Flow Matching](https://arxiv.org/abs/2302.00482)
- [PC-JeDi](https://arxiv.org/abs/2303.05376) (based on [Score-Based Models through SDEs](https://arxiv.org/abs/2011.13456))
- [PC-Droid](https://arxiv.org/abs/2307.06836) (based on [EDM Diffusion](https://arxiv.org/abs/2206.00364))
- [Flow Matching with Self-Conditioning](https://arxiv.org/abs/2310.05764) (only implemented in notebook)

### Classification Models

- [EPiC Classifier](https://arxiv.org/abs/2301.08128) (for Set-based data)
- Fully Connected Classifier

## 📊 Datasets

Click on the dataset to get more information about the dataset, the features, and how to download it.

<details>
  <summary>
    <b>JetNet</b>
  </summary>

- <b>Description:</b> ([dataset reference](https://arxiv.org/abs/2106.11535))

  - Simulated particle jets produced by proton-proton collisions in a simplified detector. The dataset is split into jets originating from tops, light quarks, gluons, W bosons, and Z bosons and has a maximum number of 150 particles per jet.

- <b>Features:</b>

  - Lightning DataModule for easy exchange of datasets
  - Preprocessing and postprocessing of data
  - Evaluation during training and after training with comet and wandb
  - Many settings for trainings (e.g. conditioning on selected features, training on muliple jet types, etc.)

- <b>Download</b>
  Can be downloaded from [Zenodo](https://zenodo.org/records/6975118) and should be saved under `data_folder_specified_in_env/jetnet/`

</details>

<details>
  <summary>
    <b>LHC Olympics</b>
  </summary>

- <b>Description:</b> ([dataset reference](https://lhco2020.github.io/homepage/))

  - A dataset for Anomaly Detection, where the generative models are used to generate the Standard Model background. It consists of 1M QCD simulated dijet events that after clustering result in 2 jets per event with up to 279 particles per jet.

- <b>Features:</b>

  - Lightning DataModule for easy exchange of datasets
  - Preprocessing and postprocessing of data
  - Evaluation during training and after training with comet and wandb
  - Many settings for trainings (e.g. conditioning on selected features, training separately on dijets, on both dijety, on the whole event, etc.)

- <b>Download and Preprocessing</b>
  Can be downloaded from [Zenodo](https://zenodo.org/records/6466204). The  `events_anomalydetection_v2.h5` is the file needed as it contains all the particles from an event. Before using, the events need to be clustered and brought into point cloud format. This preprocessing can be done with this [Code](https://github.com/ewencedr/FastJet-LHCO).
  The `events_anomalydetection_v2.h5` and the preprocessed data should be saved under `your_spedata_folder_specified_in_env/lhco`

</details>

<details>
  <summary>
    <b>JetClass</b>
  </summary>

- <b>Description:</b> ([dataset reference](https://arxiv.org/abs/2202.03772))

  - Simulated particle jets like in JetNet, but  JetClass provided much more data, more jet types and more particle features.

- <b>Features:</b>

  - Lightning DataModule for easy exchange of datasets
  - Preprocessing and postprocessing of data
  - Evaluation during training and after training with comet and wandb
  - Many settings for trainings (e.g. conditioning on selected features, training on muliple jet types, etc.)

- <b>Download and Preprocessing</b>
  Can be downloaded from [Zenodo](https://zenodo.org/records/6619768).

</details>
<details>
  <summary>
    <b>CaloChallenge?</b>
  </summary>
  To be implemented
</details>
<details>
  <summary>
    <b>TwoMoons</b>
  </summary>

- <b>Description:</b> ([dataset reference](https://lhco2020.github.io/homepage/))
  - Simple toy dataset for testing the models in the notebook. Does not need to be downloaded because the dataset can be generated via a [scikit-learn function](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

</details>

## Additional Features

- Training and Evaluation of models
- Callbacks for logging and visualization
- Preprocessing and postprocessing of data
- Notebooks for quick training and evaluation
- Notebooks for tinkering with loss functions and architectures
- EMA implementation (Exponential Moving Average)
- Best Practices for coding thanks to the [Lightning-Hydra-Template](<>) with all major benefits of [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) and [Hydra](https://hydra.cc/docs/intro/) (configurations, logging, multi-gpu, datamodules for easy exchange of datasets, etc.)

## ⚡️ Quickstart

### Preconfigured Experiments

Note, that this repository was originally used for a research project and is now being adapted to be more general. The following configurations have been used in the original project and should work. However, not all combinations of architectures and loss functions have been tested, yet and some cases might create errors. Please contact me for any issues.

## Continue

The repository contains a set of preconfigured experiments that can be used to train and evaluate the models. The experiments are defined in the [configs/experiment/](configs/experiment/) folder. The experiments are defined in yaml files and can be used to train and evaluate the models. The experiments include the model, dataset, and training parameters.

\*\*\* List of experiments \*\*\*

### Build your own experiment

### Training

This is the official repository implementing the EPiC Flow Matching point cloud generative machine learning models from arxiv1111.11111.

EPiC Flow Matching is a [Continuous Normalising Flow](https://arxiv.org/abs/1806.07366) that is trained with a simulation-free approach called [Flow Matching](https://arxiv.org/abs/2210.02747). The model uses [DeepSet](https://arxiv.org/abs/1703.06114) based [EPiC layers](https://arxiv.org/abs/2301.08128) for the architecture, which allows for good scalability to high set sizes.

Additionally to the EPiC Flow Matching model, this repository also implements various other loss functions that correspond to other flow matching/diffusion-based models, like [Conditional Flow Matching](https://arxiv.org/abs/2302.00482) and [DDIM](https://arxiv.org/abs/2010.02502) based [PC-Jedi](https://arxiv.org/abs/2303.05376).

The models are tested on the [JetNet dataset](https://zenodo.org/record/6975118). The JetNet dataset is used in particle physics to test point cloud generative deep learning architectures. It consists of simulated particle jets produced by proton-proton collisions in a simplified detector. The dataset is split into jets originating from tops, light quarks, gluons, W bosons, and Z bosons and has a maximum number of 150 particles per jet.

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
  EPiC Flow Matching trained on top30 dataset with no additional conditioning. Jet size conditioning is a necessity for the architecture
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
  EPiC Flow Matching trained on top150 dataset with no additional conditioning. Jet size conditioning is a necessity for the architecture
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
  EPiC Jedi (DDIM diffusion based) trained on top30 dataset with no additional conditioning. Jet size conditioning is a necessity for the architecture
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
  EPiC Jedi (DDIM diffusion based) trained on top150 dataset with no additional conditioning. Jet size conditioning is a necessity for the architecture
</details>
<details>
  <summary>
    <b>diffusion_alljet150_cond</b>
  </summary>
  EPiC Jedi (DDIM diffusion based) trained on all jet types with a maximum of 150 particles per jet and conditioning on jet mass and pt.
</details>
<br>
<details>
  <summary>
    <b>lhco/bigPC</b>
  </summary>
  In this training, the two dijet events of the LHCO dataset are put into one large point cloud, i.e. a point cloud of size 558 (2*279). This is more difficult for the model to learn and a clustering after training is also necessary to get the two dijet events back. However, the generation also works well and this shows that a model can learn large point clouds with less restrictions.
</details>
<details>
  <summary>
    <b>lhco/bigPC</b>
  </summary>
  In this training, the two dijet events of the LHCO dataset are put into one large point cloud, i.e. a point cloud of size 558 (2*279).
</details>

During training and evaluation, metrics and plots can be logged via comet and wandb. After training the model will be evaluated automatically and the final results will be saved locally and logged via the selected loggers. The evaluation can also be manually started like this

```bash
python src/eval.py experiment=experiment_name.yaml ckpt_path=checkpoint_path
```

You can also specify the config file that was saved at the beginning of the training

```bash
python src/eval.py cfg_path=<cfg_file_path> ckpt_path=<checkpoint_path>
```

Notebooks are available to quickly train, evaluate models and create plots.

## Contributing

Please feel free to contribute to this repository. If you have any questions, feel free to open an issue or contact me directly. When contributing, please make sure to follow style guidelines specified in the pre-commit hooks.

## Note of Caution

This repository was originally used for a research project and is now being adapted to be more general. The preconfigured use cases have been tested and should work. However, some code might be a bit specific to a certain use case and could be generalized further to allow for more flexibility. Please contact me for any issues.

TODO

- [ ] Context Normaliser should be deleted

- [ ] create separate preprocessing class/ change all datamodels to scipy preprocessing

- [ ] does transformer work?

- model folder is okay

## Citation

When using this repository in research, please cite the original papers:

```bibtex
@misc{birk2023flow,
      title={Flow Matching Beyond Kinematics: Generating Jets with Particle-ID and Trajectory Displacement Information},
      author={Joschka Birk and Erik Buhmann and Cedric Ewen and Gregor Kasieczka and David Shih},
      year={2023},
      eprint={2312.00123},
      archivePrefix={arXiv},
      primaryClass={hep-ph}
}
```

```bibtex
@misc{buhmann2023phase,
      title={Full Phase Space Resonant Anomaly Detection},
      author={Erik Buhmann and Cedric Ewen and Gregor Kasieczka and Vinicius Mikuni and Benjamin Nachman and David Shih},
      year={2023},
      eprint={2310.06897},
      archivePrefix={arXiv},
      primaryClass={hep-ph}
}
```

```bibtex
@misc{buhmann2023epicly,
      title={EPiC-ly Fast Particle Cloud Generation with Flow-Matching and Diffusion},
      author={Erik Buhmann and Cedric Ewen and Darius A. Faroughy and Tobias Golling and Gregor Kasieczka and Matthew Leigh and Guillaume Quétant and John Andrew Raine and Debajyoti Sengupta and David Shih},
      year={2023},
      eprint={2310.00049},
      archivePrefix={arXiv},
      primaryClass={hep-ph}
}
```
