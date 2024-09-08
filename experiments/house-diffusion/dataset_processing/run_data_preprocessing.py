

from dataclasses import dataclass
import json
import os
import pickle

from omegaconf import OmegaConf
from tqdm import tqdm

from data_preprocessing.process_structure import process_structure, process_structure_alternative_method

from data_preprocessing.process_graphs import housegandatareader_like_process_subgraph, housegandatareader_like_process_subgraph_test_sample

from data_preprocessing.raw_data_sample import DataSample
from data_preprocessing.generate_housediffusion_features import HouseDiffusionSampleGenerator

import dataclasses


@dataclass
class MyConfig:
    """OmegaConf config for data preprocessing options.
    
    Args:
        datapath (str): Path to the folder containing the struct_in/struct_out and graph_in/graph_out folders.
        is_train (bool): Whether to process the train or test set.

        use_topojson_processing (bool): Whether to use alternative method for processing structural image that should result in fewer wall lines.
        name (str): Name of the output folder.

        make_mrr (bool): Whether to use MRR (rectangle / 4 corner) approximation for rooms.

        room_type_dim (int): Length of the one-hot encoding of room types.
        corner_index_dim (int): Length of the one-hot encoding of corner indices.
        room_index_dim (int): Length of the one-hot encoding of room indices.
"""

    datapath: str = "path/to/modified-swiss-dwellings-v2"
    is_train: bool = True

    use_topojson_processing: bool = True

    name: str = "house_dicts_with_topojson_processing"

    make_mrr: bool = True

    room_type_dim: int = 15
    corner_index_dim: int = 4
    room_index_dim: int = 96

def gather_ids(datapath, sub_path="struct_in"):
    ids = []

    for id in os.listdir(os.path.join(datapath, sub_path)):
        ids.append(int(id.split(".")[0]))

    return sorted(ids)


def process_sample(house_diffusion_sample_generator: HouseDiffusionSampleGenerator, sample: DataSample, make_mrr):
    if conf.use_topojson_processing:
        structure = process_structure_alternative_method(sample.structural_img)
    else:
        structure, _ = process_structure(sample.structural_img)

    if sample.is_train:
        rooms, rooms_graph = housegandatareader_like_process_subgraph(sample.graph_out, sample.image_bounds, sample.out_img, make_mrr=make_mrr)
    else:
        override_num_corners = 4 if make_mrr else None
        rooms, rooms_graph = housegandatareader_like_process_subgraph_test_sample(sample.graph_pred, override_num_corners)

    house_sample = house_diffusion_sample_generator.generate_house_diffusion_feats(rooms, structure)

    house_dict = dataclasses.asdict(house_sample)

    if house_sample.number_of_corners > 350:
        raise ValueError(f"House {sample.id} has more than 350 corners")

    return house_dict



if __name__ == "__main__":
    conf: MyConfig = OmegaConf.structured(MyConfig)

    conf.merge_with_cli()

    print(conf)

    name = conf.name

    # house_dicts = []

    failed_ids = {}

    os.makedirs(name, exist_ok=True)

    house_diffusion_sample_generator = HouseDiffusionSampleGenerator(room_type_dim=conf.room_type_dim, corner_index_dim=conf.corner_index_dim, room_index_dim=conf.room_index_dim)

    for id in tqdm(gather_ids(conf.datapath)):
        try:
            datasample = DataSample(conf.datapath, id, is_train=conf.is_train, use_MRR=conf.make_mrr)

            house_dict = process_sample(house_diffusion_sample_generator, datasample, make_mrr=conf.make_mrr)

            # house_dicts.append(house_dict)

            assert isinstance(house_dict, dict)

            with open(f"{name}/{id}.pickle", "wb") as f:
                pickle.dump(house_dict, f)

            # print(f"Successfully processed {id}")
        except ValueError as e:
            print(f"Failed to process {id}")
            failed_ids[id] = str(e)

    with open(f"{name}_failed_ids.json", "w") as f:
        json.dump(failed_ids, f)

    # np.savez_compressed(f"{name}.npz", house_dicts=house_dicts)