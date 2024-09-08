# Modified Swiss Dwellings preprocessing for HouseDiffusion

This repository is based on [cvaad-workshop/iccv23-challenge](https://github.com/cvaad-workshop/iccv23-challenge). It contains a script for preprocessing the Modified Swiss Dwellings dataset to use in an adapted version of HouseDiffusion with structure walls as additional input.

## Conda envirnment

A conda environment that should also contain the packages needed for running the preprocessing is provided in `pytorchhouseddifusion.yml` file.

## How to run preprocessing steps:

The file you should run for preprocessing is: [`run_data_preprocessing.py`](run_data_preprocessing.py). The options for running are set through OmegaConf, and can be specified on the command line. See the `MyConfig` class in [`run_data_preprocessing.py`](run_data_preprocessing.py) for all the options that can be specified.

### Preprocess training examples

Generate training samples with the Minimum Rotated Rectangle approximation (MRR):

```python run_data_preprocessing.py datapath='path_to_datasets/modified-swiss-dwellings/modified-swiss-dwellings-v1-train' is_train=true use_topojson_processing=true name=house_dicts_mrr make_mrr=true```

Generate training samples without MRR approximation:

```python run_data_preprocessing.py datapath='path_to_datasets/modified-swiss-dwellings/modified-swiss-dwellings-v1-train' is_train=true use_topojson_processing=true name=house_dicts_full make_mrr=false```

### Preprocess test samples

To generate test samples, the preprocessing code expects additional folders to be available in the `modified-swiss-dwellings-v1-test` folder. If `make_mrr=true`, the folder `graph_pred` should exists, and if `make_mrr=false` the folder `graph_pred_n_corners` should exist. The files in this folder can be derived from the `graph_in` files. The graphs in `graph_pred` have an additional key, `room_type`, and `graph_pred_n_corners` have two additional keys: `room_type` and `n_corners`, the number of corners the room should have.

The room_type can be predicted using a graph neural network (GNN) that predicts the room_type from the zoning type and graph structure. Use the [`node_classification_room_type.ipynb`](node_classification_room_type.ipynb) notebook to train a GNN, and save predicted room types for the test set. Likewise use the [`node_classification_number_of_corners.ipynb`](node_classification_number_of_corners.ipynb) in addition to the [`node_classification_room_type.ipynb`](node_classification_room_type.ipynb) notebook. to predict how many corners a room should have in case of not using MRR.

After using the notebook to create the `graph_pred` and/or `graph_pred_n_corners` folder, the test house_dicts can be generated:

Generate test samples with the Minimum Rotated Rectangle approximation (MRR):
```python run_data_preprocessing.py datapath='path_to_datasets/modified-swiss-dwellings/modified-swiss-dwellings-v1-test' is_train=false use_topojson_processing=true name=house_dicts_mrr_test make_mrr=true```


### How to turn preprocessing output into expected dataset format

The output of the preprocessing script is a folder of pickled dictionaries (house_dicts) that contain the features needed for training. They are created converting `HouseDiffusionSample` dataclass object to a dictionary. To turn the folder of house_dicts into the expected dataset format for training it should be wrapped in a folder that names the dataset, and contains train_ids.csv, and val_ids.csv in addition to the house_dicts folder:

```markdown
├── datasets
│   ├── modified_swiss_dwellings
|   |   └── house_dicts
|   |   |   └── 0.pickle
|   |   |   └── 1.pickle
|   |   |   └── ...
|   |   └── train_ids.csv
|   |   └── val_ids.csv
```

## MRR Approximation

Enabling this option finds the minimum rotated rectangle for each room polygon, and saves these rectangle polygons instead of the original room polygon.

## Without MRR Approximation

The original room corners are still not used directly, as some rooms have a lot more corners than expected. Instead the polygons are first simplified with a small tolerance, to remove extraneous corners. See the method: [`turn_polygon_into_house`](turn_polygon_into_house/process_graphs.py)