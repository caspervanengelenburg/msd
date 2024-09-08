import os
import pickle
import typing
import pandas as pd
import torch
from torch.utils.data import Dataset

import random

import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate

from house_diffusion.house_diffusion_sample import HouseDiffusionSample

DEFAULT_DATASET_PATH = "modified_swiss_dwellings_topojson_processing"

def get_dataloader_modified_swiss_dwellings(
    batch_size,
    set_name = 'train',
    override_use_augmentation=None,
    override_shuffle=None,
    dataset_name=DEFAULT_DATASET_PATH,
    ids_list=None,
    use_structural_feats=True,
) -> typing.Iterator[typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"Loading modified swiss dwellings from ")

    print(f"Using dataset path: {dataset_name}")
    
    is_train_set = set_name=='train'

    use_augmentation = is_train_set
    shuffle = is_train_set

    if override_use_augmentation is not None:
        use_augmentation = override_use_augmentation
    
    if override_shuffle is not None:
        shuffle = override_shuffle

    if ids_list is not None:
        dataset = ModifiedSwissDwellingsDataset(f"../datasets/{dataset_name}/house_dicts", 
                                                ids_csv=None,
                                                use_augmentation=use_augmentation,
                                                ids_list=ids_list,
                                                use_structural_feats=use_structural_feats)
    else:
        dataset = ModifiedSwissDwellingsDataset(f"../datasets/{dataset_name}/house_dicts", 
                                                f"../datasets/{dataset_name}/{set_name}_ids.csv", 
                                                use_augmentation=use_augmentation,
                                                use_structural_feats=use_structural_feats)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=False, collate_fn=dataset.collate_house_diffusion_samples
    )

    return loader



def load_modified_swiss_dwellings(
    batch_size,
    set_name = 'train',
    override_use_augmentation=None,
    override_shuffle=None,
    dataset_name=DEFAULT_DATASET_PATH,
    ids_list=None,
    use_structural_feats=True,
) -> typing.Iterator[typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    
    loader = get_dataloader_modified_swiss_dwellings(batch_size, set_name, override_use_augmentation, override_shuffle, dataset_name=dataset_name, ids_list=ids_list, use_structural_feats=use_structural_feats)

    while True:
        yield from loader


class ModifiedSwissDwellingsDataset(Dataset):
    def __init__(self, data_path, ids_csv, use_augmentation=True, ids_list=None, use_structural_feats=True):

        if ids_csv is not None:
            self.ids = pd.read_csv(ids_csv, header=None).values.flatten().tolist()

            print("Using ids from ids csv")
        elif ids_list is not None:
            self.ids = ids_list

            print("Using ids from ids list")
        else:
            self.ids = gather_ids(data_path)

            print("Using all ids from folder")

        print(f"ModifiedSwissDwellingsDataset: {ids_csv=}; {use_augmentation=}; {len(self.ids)=}")

        houses = [load_id(data_path, id) for id in self.ids]

        self.houses = [HouseDiffusionSample(**house) for house in houses]

        self.use_augmentation = use_augmentation

        self.use_structural_feats = use_structural_feats

    def __len__(self):
        return len(self.houses)

    def __getitem__(self, idx) -> HouseDiffusionSample:
        """Returns a HouseDiffusionSample."""

        house: HouseDiffusionSample = self.houses[idx]

        house.id = self.ids[idx]

        # The modified swiss dwellings dataset had geometries stored as (y, x) instead of (x, y) as would be expected
        house = house.with_xy_corners_swapped()


        # Make features between -0.5 and 0.5
        house = house.with_normalized_geometry()

        if self.use_augmentation:
            house = random_augment_house(house)

        return house
    
    def collate_house_diffusion_samples(self, samples: typing.List[HouseDiffusionSample]):
        max_corners = max([sample.corners.shape[0] for sample in samples]) + 1
        max_struct_corners = max([sample.struct_corners_a.shape[0] for sample in samples]) + 1

        samples_with_feats = [sample.get_feats(max_corners, max_struct_corners, transpose_geometries=True, return_structural_feats=self.use_structural_feats) for sample in samples]

        collated = default_collate(samples_with_feats)

        corners = collated["corners"]

        del collated["corners"]

        return corners, collated



def gather_ids(datapath):
    ids = []

    for id in os.listdir(datapath):
        ids.append(int(id.split(".")[0]))

    return sorted(ids)


def load_id(datapath, id):
    with open(os.path.join(datapath, f"{id}.pickle"), "rb") as f:
        house_dict = pickle.load(f)
    
    return house_dict



# Geometry augmentation:
def rotate_right_angle(arr, rotation):
    """This actually also does flips?"""

    if rotation == 1:
        arr[:, [0, 1]] = arr[:, [1, 0]]
        arr[:, 0] = -arr[:, 0]
    elif rotation == 2:
        arr[:, [0, 1]] = -arr[:, [1, 0]]
    elif rotation == 3:
        arr[:, [0, 1]] = arr[:, [1, 0]]
        arr[:, 1] = -arr[:, 1]
    
    return arr


def random_augment_rotate_right_angle(arrays: typing.List[np.ndarray]):
    # Sample random rotation
    rotation = random.randint(0,3)

    arrays = [rotate_right_angle(arr, rotation) for arr in arrays]    

    return arrays


def apply_rotation(rotation_matrix, arr):
    assert rotation_matrix.shape == (2, 2)
    assert arr.shape[1] == 2

    return np.matmul(rotation_matrix, arr.transpose([1, 0])).transpose([1, 0])


def random_rotate(arrays: typing.List[np.ndarray], factor = 1.0):

    max_rotation = factor * 2 * np.pi

    # Sample random rotation
    rotation = random.uniform(-max_rotation, max_rotation)

    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])

    arrays = [apply_rotation(rotation_matrix, arr) for arr in arrays]

    return arrays

def random_shift(arrays: typing.List[np.ndarray]):
    shift = np.random.normal(0., .01, size=2)

    arrays = [arr + shift for arr in arrays]

    return arrays

def random_scale(arrays: typing.List[np.ndarray]):
    scale = np.random.normal(1., .1)

    arrays = [arr * scale for arr in arrays]

    return arrays


number_of_walls_to_remove = {
    0: 0.75,
    1: 0.125,
    2: 0.075,
    3: 0.025,
    4: 0.025,
}

def random_augment_house(house: HouseDiffusionSample):
    house = house.with_geometric_augmentation(random_augment_rotate_right_angle)
    
    if np.random.rand() < 0.75:
        house = house.with_geometric_augmentation(random_rotate)
    
    if np.random.rand() < 0.5:
        house = house.with_geometric_augmentation(random_shift)
    
    if np.random.rand() < 0.5:
        house = house.with_geometric_augmentation(random_scale)
    
    for _ in range(np.random.choice(list(number_of_walls_to_remove.keys()), p=list(number_of_walls_to_remove.values()))):
        house = house.with_wall_i_masked_out(np.random.randint(0, house.number_of_walls))
    
    house = house.with_randomized_room_indices()

    house = house.with_randomized_wall_point_order()
    
    return house
