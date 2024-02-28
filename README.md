<div align="center">

# Particle-FM/Diffusion

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/) <br>
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

[![arxiv](http://img.shields.io/badge/arXiv-2310.00049-B31B1B.svg)](https://arxiv.org/abs/2310.00049)
[![arxiv](http://img.shields.io/badge/arXiv-2310.06897-B31B1B.svg)](https://arxiv.org/abs/2310.06897)
[![Paper](http://img.shields.io/badge/arxiv-2312.00123-B31B1B.svg)](https://arxiv.org/abs/2312.00123)

</div>

## ℹ️ Description

This repository contains multiple (mostly) generative neural networks and multiple datasets from particle physics. The focus lies on continuous time generative models for point clouds. Diffusion/ Score-based models / Continuous Normalizing Flows (CNFs) are in this repository combined under the [Flow Matching](https://arxiv.org/abs/2210.02747) (FM) framework, where these models differ only in the loss function.

This repository contains code for the following papers as well as additional models and datasets. For the code repository that only contain the code for the papers, please refer to the repositories linked in the papers.
- [EPiC-ly Fast Particle Cloud Generation with Flow-Matching and Diffusion](https://arxiv.org/abs/2310.00049); Erik Buhmann, Cedric Ewen, Darius A. Faroughy, Tobias Golling, Gregor Kasieczka, Matthew Leigh, Guillaume Quétant, John Andrew Raine, Debajyoti Sengupta, David Shih; 2023; [Code](https://github.com/uhh-pd-ml/EPiC-FM)
- [Full Phase Space Resonant Anomaly Detection](https://arxiv.org/abs/2310.06897); Erik Buhmann, Cedric Ewen, Gregor Kasieczka, Vinicius Mikuni, Benjamin Nachman, David Shih; 2023; [Code](https://github.com/uhh-pd-ml/LHCO_EPiC-FM)
- [Flow Matching Beyond Kinematics: Generating Jets with Particle-ID and Trajectory Displacement Information](https://arxiv.org/abs/2312.00123); Joschka Birk, Erik Buhmann, Cedric Ewen, Gregor Kasieczka, David Shih; 2023; [Code](https://github.com/uhh-pd-ml/beyond_kinematics) 

## 🤖 Models

### Generative Models

#### Architectures:

- [EPiC](https://arxiv.org/abs/2301.08128) (for Sets, based on [DeepSets](https://arxiv.org/abs/1703.06114))
- [Full Transformer](https://arxiv.org/abs/2307.06836) (for Sets)
- [CrossAttention Transformer](https://arxiv.org/abs/2307.06836) (for Sets)
- [MDMA](https://arxiv.org/abs/2305.15254) (for Sets, CrossAttention Transformer)
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
  Can be downloaded from [Zenodo](https://zenodo.org/records/6619768) by following the instructions from [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer).
  Adjust the paths in the `configs/preprocessing/data.yaml` and run 

  ```bash
  python scripts/prepare_dataset.py && python scripts/preprocessing.py
  ```

</details>
<details>
  <summary>
    <b>CaloChallenge</b>
  </summary>
  TBD
</details>
<details>
  <summary>
    <b>TwoMoons</b>
  </summary>

- <b>Description:</b> ([dataset reference](https://lhco2020.github.io/homepage/))
  - Simple toy dataset for testing the models in the notebook. Does not need to be downloaded because the dataset can be generated via a [scikit-learn function](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

</details>

## 🌟 Additional Features

- Training and Evaluation of models
- Callbacks for logging and visualization
- Preprocessing and postprocessing of data
- Notebooks for quick training and evaluation
- Notebooks for tinkering with loss functions and architectures
- EMA implementation (Exponential Moving Average)
- Best Practices for coding thanks to the [Lightning-Hydra-Template](<>) with all major benefits of [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) and [Hydra](https://hydra.cc/docs/intro/) (configurations, logging, multi-gpu, datamodules for easy exchange of datasets, etc.)

## ⚡️ Quickstart

### ⚙️ Installation

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

### 🧠 Training

Train model with default configuration

```bash
# train on one GPU
python src/train.py trainer=gpu

# train on multiple GPUs
python src/train.py trainer=ddp
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

### 📈 Evaluation

During training and evaluation, metrics and plots are automatically evaluated via custom lightning callbacks and logged via comet and wandb. After training most models will also be evaluated automatically and the final results will be saved locally and logged via the selected loggers. The evaluation can also be manually started like this

```bash
python src/eval.py experiment=experiment_name.yaml ckpt_path=checkpoint_path
```

You can also specify the config file that was saved at the beginning of the training

```bash
python src/eval.py cfg_path=<cfg_file_path> ckpt_path=<checkpoint_path>
```

Notebooks are available to quickly train, evaluate models and create plots.


## 🚀 Preconfigured Experiments

The experiments are defined in yaml files and specify which loss function and architecture to use, which dataset to use, and which hyperparameters to use. Feel free to create your own experiments, but some preconfigured experiments are available in the [configs/experiment/](configs/experiment/) folder. 

<b>Click on the dataset names</b> to find out more about all the available experiments for the dataset.


<details>
  <summary>
    <b>JetNet Dataset</b>
  </summary>
  For the JetNet dataset, the experiments from the paper [2310.00049](https://arxiv.org/abs/2310.00049) are available: 

  - `fm_tops30_cond`, `fm_tops30`, `fm_tops150_cond`, `fm_tops150`, where all are EPiC Flow Matching models trained on the top dataset. The numbers indicate whether the model is trained on the top30 or top150 dataset and the `_cond` indicates that the model is conditioned on the jet mass and pt.
  - `diffusion_tops30_cond`, `diffusion_tops30`, `diffusion_tops150_cond`, `diffusion_tops150`, where all are EPiC-JeDi models trained on the top dataset. The numbers indicate whether the model is trained on the top30 or top150 dataset and the `_cond` indicates that the model is conditioned on the jet mass and pt.

  Although not shown in the paper, the models can easily be trained on different combinations of jet types and jet sizes. Examples are:
  - `fm_alljet150_cond`, which is an EPiC Flow Matching model trained on all jet types with a maximum of 150 particles per jet and conditioning on jet mass and pt.
  - `diffusion_alljet150_cond`, which is an EPiC-JeDi model trained on all jet types with a maximum of 150 particles per jet and conditioning on jet mass and pt.

  Additionally, other architectures can be used:
  - `fm_mdma`, which is an EPiC Flow Matching model trained on the top dataset with the MDMA architecture
</details>


<details>
  <summary>
    <b>LHCO Dataset</b>
  </summary>
The LHCO dataset consists of two dijets per event, which allows for multiple ways of generating these two jets. By clustering the event into two jets, both jets can be seen as a single point cloud similarly to the JetNet dataset. Using this clustering, the following experiments are available for the point cloud models:


- `lhco/both_jets`  One EPiC-FM model trained on point clouds of jet 1 and jet 2 at the same time (experiment from the paper [2310.06897](https://arxiv.org/abs/2310.06897)) 
- `lhco/x_jet` / `lhco_y_jet` One EPiC-FM model trained on one point cloud, where `lhco/x_jet` trains the model on jet 1 and `lhco/y_jet` trains the model on jet 2
- `lhco/jets_crossattention` Same as `lhco/both_jets` but with a crossattention transformer from [2307.06836](https://arxiv.org/abs/2307.06836) instead of the EPiC architecture
- `lhco/transformer` Same as `lhco/both_jets` but with a full transformer from [2307.06836](https://arxiv.org/abs/2307.06836) instead of the EPiC architecture

All these models require a conditioning on the jet features of the full dijet event:

- `lhco/jet_features` FM-Model with fully connected architecture trained on the jet features of both dijets to condition the generation of the point clouds (experiment from the paper [2310.06897](https://arxiv.org/abs/2310.06897))


Instead of the two step approach, the event can also be generated in more complex ways:

- `lhco/bigPC` Both point clouds of the dijets are put into one large point cloud and the model is trained on this large point cloud. In the evaluation, the point cloud is clustered into two jets again
- `lhco/wholeEvent` The generative model can also directly be trained on the whole event, which is more difficult for the model to learn and a clustering for evaluation is also necessary. However, this still works well and shows that these models are powerful enough to learn large point clouds with less restrictions.


Additionally, <b>classifiers</b> are available to check, if the generated events are distinguishable from the original events.

- `lhco/epic_classifier` point cloud classifier based on the EPiC architecture. Paths to data must be specified in the config file.
- `lhco/hl_classifier` fully connected classifier as in [2109.00546](https://arxiv.org/abs/2109.00546) to compare high level features. Paths to data must be specified in the config file.
</details>
<details>
  <summary>
    <b>JetClass Dataset</b>
  </summary>
  
  - `jetclass/jetclass_cond` EPiC Flow Matching model trained on the JetClass dataset and conditioned
  - `jetclass/jetclass_classifier` After evaluating the generative model, a classifier test can be run. For this, the paths to the generated data needs to be specified in the config file.
</details>
<details>
  <summary>
    <b>CaloChallenge Dataset</b>
  </summary>
  TBD
</details>

## 🫱🏼‍🫲🏽 Contributing

Please feel free to contribute to this repository. If you have any questions, feel free to open an issue or contact me directly. When contributing, please make sure to follow style guidelines by using the pre-commit hooks.

## ⚠ Note of Caution

This repository was originally used for a research project and is now being adapted to be more general. The experiments published in the papers have been tested and should work. All other preconfigured experiments should work but might have some minor issues and some configurations of datasets/models might not work out of the box. Some code might be a bit specific to a certain use case and could be generalized further to allow for more flexibility. Additionally, some leftovers from development might also still exist that don't work. Please create an issue if you encounter any problems or have any questions.

## 📚 Citation

When using this repository in research, please cite the following papers:

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


## TODOs

- [ ] Context Normaliser should be deleted

- [ ] create separate preprocessing class/ change all datamodels to scipy preprocessing

- [ ] does transformer work?

- [ ] clean up notebooks
- [ ] add dataset
- [ ] setup file for installation
