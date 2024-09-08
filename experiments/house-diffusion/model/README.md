# HouseDiffusion adapted for Modified Swiss Dwellings dataset

This repository is a fork of the original [HouseDiffusion repository](https://github.com/aminshabani/house_diffusion.git).

**[HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising](https://arxiv.org/abs/2211.13287)**
<img src='figs/teaser.png' width=100%>

**Note from orignal [HouseDiffusion repository](https://github.com/aminshabani/house_diffusion.git):**


> Our implementation is based on the public implementation of [guided-diffusion](https://github.com/openai/guided-diffusion). For installation instructions, please refer to their repository. Keep in mind that our current version has not been cleaned and some features from the original repository may not function correctly.


## 1. Installation
**1. Clone the repo and install the requirements:**

It's recommended to install the required packages through conda/mamba. An environment file is provided at `pytorchhousediffusion.yml`. After installing, use pip install -e to install the house_diffusion folder/package:


```
cd house_diffusion

mamba install -f pytorchhousediffusion.yml
mamba activate pytorchhousediffusion

pip install -e .
```

Wandb is used for logging.


**2. Download the dataset and create the datasets directory**

- Use the preprocessing scripts in [dataset_processing](../dataset_processing/README.md) to generate house_dict folders. Then place the house_dict folder in a modified_swiss_dwellings folder, along side train_ids.csv and val_ids.csv. The csv files contain only a list of id numbers per line and determine the train and validation split. 

You might want to filter the train_ids, on floorplans that have at most N corners (for some N), to exclude floorplans with more than N corners from training. There are two reasons for filtering floorplans with large numbers of corners: (1) without excluding them the one hot encoded room id vector will get very long only to support a few more training examples, and (2) to limit the maximum amount of GPU memory needed per batch.

```
house_diffusion
├── datasets
│   ├── modified_swiss_dwellings
|   |   └── house_dicts
|   |   |   └── 0.pickle
|   |   |   └── 1.pickle
|   |   |   └── ...
|   |   └── train_ids.csv
|   |   └── val_ids.csv
└── guided_diffusion
└── scripts
└── ...
```


## 2. Training

First navigate to the scripts directory:
```
cd scripts
```


You can run a single experiment using the following command:

```
python image_train.py --dataset modified_swiss_dwellings --batch_size 32 --set_name train --timeout 36:00:00 --save_interval 2000 --test_interval 1000 --use_wall_self_attention true
```

### Options:

#### --dataset

Specifies which dataset folder to load from the `datasets/` directory. 

In addition `--dataset` also controls the amount of input and condition channels, see [`script_util.py`](house_diffusion/script_util.py).

The supported modified swiss dwellings options are: 'modified_swiss_dwellings', 'modified_swiss_dwellings_without_structural', and 'modified_swiss_dwellings_all_corners'. See [Supported --dataset options](#supported---dataset-options).

Support for additional dataset variations can be added in `update_arg_parser` in [script_util.py](house_diffusion/script_util.py).

#### --batch_size

How many samples to use per batch. 

Note that the GPU usage per batch differs, because not every batch has the same number of room/structural corners.

#### --set_name

Should be "train" for training. Determines if the data is shuffled and determines the augmentations that are used.

#### --timeout
timeout specified in hh:mm:dd after which to stop.

#### --save_interval

How often a checkpoint is saved. Checkpoints are stored in `ckpts/run name/`.

#### --test_interval

How often inference examples are created and logged to wandb.

#### --use_wall_self_attention

If there should be self-attention step between the input wall corners and the cross attention step.

#### --lr

Learning rate default is: 0.0001. The lr is divided by 10 every 100k steps. When loading a checkpoint, the amount of steps of the checkpoint is added to the total steps, and total steps is used for the lr computation as: `lr = [--lr] * (0.1**(total_steps//100000))`.

#### --resume_checkpoint

Path to checkpoint to resume from.

### Supported --dataset options:

#### 'modified_swiss_dwellings' (with MRR)
'modified_swiss_dwellings' expects data to be preprocessed for 115 condition channels, by using the following preprocessing:

```bash
python run_data_preprocessing.py datapath='path_to_datasets/modified-swiss-dwellings/modified-swiss-dwellings-v1-train' is_train=true use_topojson_processing=true name=house_dicts_mrr make_mrr=true room_type_dim=15 corner_index_dim=4 room_index_dim=96```

```
Note that room_type_dim=15, corner_index_dim=4, and room_index_dim=96 indeed add up to 115 condition channels.

#### 'modified_swiss_dwellings_all_corners' (without MRR)
'modified_swiss_dwellings' expects data to be preprocessed for 183 condition channels, e.g. by using the following preprocessing:

```bash
python run_data_preprocessing.py datapath='path_to_datasets/modified-swiss-dwellings/modified-swiss-dwellings-v1-train' is_train=true use_topojson_processing=true name=house_dicts_full make_mrr=false room_type_dim=15 corner_index_dim=72 room_index_dim=96```

```

Note that room_type_dim=15, corner_index_dim=72, and room_index_dim=96 indeed add up to 183 condition channels.

#### 'modified_swiss_dwellings_without_structural' (with MRR)

Using this dataset option disable the structural cross attention. The dataset itself can have the structural features in it, they just won't be used by the model. As the script does except there to exist a folder named `modified_swiss_dwellings_without_structural`, you can fix this by symlinking to the default modified_swiss_dwellings:

```bash
cd ../datasets
ln -s modified_swiss_dwellings modified_swiss_dwellings_without_structural
cd ../scripts
```

This dataset variant uses the `EncoderLayer` as in the orignal HouseDiffusion. See the code for differences. Differences include not using different attention masks per door type.

## 3. Inference
To sample floorplans, and store resulting images (and pickles of predictions), run image_inference_msd.py:

```bash
python image_inference_msd.py --dataset modified_swiss_dwellings_all_corners --batch_size 4 --set_name test --model_path ckpts/hpc/model791000.pt --path_struct "path to modified swiss dwellings download/modified-swiss-dwellings/modified-swiss-dwellings-v1-test/struct_in/" --dataset_name modified_swiss_dwellings_all_corners_testset
```

### Options
**--path_struct** Path to the struct_in folder as downloaded from the modified swiss dwellings dataset.

**--dataset_name** The name of the dataset folder containing test inputs.

## 4. Sampling visualization from original house_diffusion repo
To provide different visualizations, please see the `save_samples` function from `scripts/image_sample.py`

Example usage:
```bash
python image_sample.py --dataset modified_swiss_dwellings --batch_size 4 --set_name val --num_samples 4 --model_path ckpts_hpc/exp/model214000.pt --use_wall_self_attention true --save_gif true
```


## Citation

```
@article{shabani2022housediffusion,
  title={HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising},
  author={Shabani, Mohammad Amin and Hosseini, Sepidehsadat and Furukawa, Yasutaka},
  journal={arXiv preprint arXiv:2211.13287},
  year={2022}
}
```
