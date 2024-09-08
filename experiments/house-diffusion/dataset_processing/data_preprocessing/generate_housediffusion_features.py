from dataclasses import dataclass
import typing
import numpy as np
import pandas as pd

from .process_structure import turn_walls_into_single_lines

from shapely import geometry

from house_diffusion_sample import HouseDiffusionSample


def get_one_hot(values: np.array, dim):
    return np.eye(dim)[values]

class HouseDiffusionSampleGenerator:
    def __init__(self,
                    corner_dim: int = 2,
                    room_type_dim: int = 15,
                    corner_index_dim: int = 4,
                    room_index_dim: int = 96,

                    padding_mask_dim: int = 1,
                    connection_dim: int = 2,

                    all_connectivity_types: typing.List[str] = ["door", "passage", "entrance"],
                 ) -> None:
        
        self.corner_dim = corner_dim
        self.room_type_dim = room_type_dim
        self.corner_index_dim = corner_index_dim
        self.room_index_dim = room_index_dim

        self.padding_mask_dim = padding_mask_dim
        self.connection_dim = connection_dim

        self.all_connectivity_types = all_connectivity_types


    def encode_room_type(self, room_type: int):
        assert room_type >= 0 and room_type < self.room_type_dim

        return get_one_hot(room_type, self.room_type_dim)

    def encode_corner_index(self, corner_index: int):
        try:
            assert corner_index >= 0 and corner_index < self.corner_index_dim, f"Corner index {corner_index} is not in the range [0, {self.corner_index_dim})"
        except AssertionError as e:
            raise ValueError(e)

        return get_one_hot(corner_index, self.corner_index_dim)

    def encode_room_index(self, room_index: int):
        if room_index >= self.room_index_dim:
            raise ValueError(f"Room index {room_index} is greater than ROOM_INDEX_DIM {self.room_index_dim}")

        assert room_index >= 0 and room_index < self.room_index_dim, f"Room index {room_index} is not in the range [0, {self.room_index_dim})"

        return get_one_hot(room_index, self.room_index_dim)


    # Attention mask for Component-wise Self Attention in the paper?
    @staticmethod
    def generate_self_mask(room_corner_bounds, max_num_points):
        
        # Intialize to mask out all corners
        self_mask = np.ones((max_num_points, max_num_points))

        # Set entries corresponding to corners in the same room to 0
        for start, end in room_corner_bounds:
            self_mask[start:end, start:end] = 0
        
        return self_mask

    # Attention mask for Global Self Attention in the paper?
    @staticmethod
    def generate_gen_mask(number_of_points_in_layout, max_number_points):
        assert number_of_points_in_layout <= max_number_points

        gen_mask = np.ones((max_number_points, max_number_points))
        gen_mask[:number_of_points_in_layout, :number_of_points_in_layout] = 0

        return gen_mask

    # Attention mask for Relational Cross Attention in the paper?
    def generate_door_mask(self, graph_edges, corner_bounds, max_number_points, target_connectivity_types=None):
        """
        
        :param target_connectivity_types: list of connectivity types to mask out. If None, self.all_connectivity_types is used.
        """

        if target_connectivity_types is None:
            target_connectivity_types = self.all_connectivity_types
        
        # Intialize to mask out all corners
        door_mask = np.ones((max_number_points, max_number_points))

        # For each pair of rooms that have a connection, set all the corners in those rooms to 0
        for (u, v), connectivity in graph_edges.items():
            assert connectivity in self.all_connectivity_types

            assert (v, u) in graph_edges, "Graph edges should be symmetric"

            if connectivity not in target_connectivity_types:
                continue

            u_start, u_end = corner_bounds[u]
            v_start, v_end = corner_bounds[v]

            door_mask[u_start:u_end, v_start:v_end] = 0
        
        return door_mask

    @staticmethod
    def generate_struct_mask(number_of_points_in_layout, number_of_walls):

        max_number_points, max_number_walls = number_of_points_in_layout, number_of_walls
        
        # Mask out all pairs of points
        struct_mask = np.ones((max_number_points, max_number_walls))

        # Set entries for pairs of existing points and walls to 0
        # (yes, because of how max_number_points and max_number_walls are defined, this is the same as setting all entries to 0)
        struct_mask[:number_of_points_in_layout, :number_of_walls] = 0


        return struct_mask

    def _generate_rooms_feats(self, house: typing.List[typing.Tuple[np.ndarray, int]], room_indices: typing.List[int]):
        num_points = 0
        room_corner_bounds = []
        rooms = []

        for (corners, room_type), r_index in zip(house, room_indices):

            num_room_corners = len(corners)

            rtype = np.repeat(np.array([self.encode_room_type(room_type)]), num_room_corners, axis=0)
            room_index = np.repeat(np.array([self.encode_room_index(r_index)]), num_room_corners, axis=0)

            corner_index = np.array([self.encode_corner_index(x) for x in range(num_room_corners)])


            # Indicates in the feature vector that this is an actual corner
            padding_mask = np.repeat([[1]], num_room_corners, axis=0)

            connections = np.array([[i, (i+1)%num_room_corners] for i in range(num_room_corners)])
            connections += num_points

            # Keep track of the [first, last) indices of the corners of each room
            room_corner_bounds.append([num_points, num_points + num_room_corners])

            num_points += num_room_corners

            # # N x F matrix where N is the number of corners (4 for a rotated rectangle) and F is the number of features per corner
            # room_matrix = np.concatenate([corners, rtype, corner_index, room_index, padding_mask, connections], axis=1, )

            rooms.append({
                "corners": corners,
                "room_types": rtype,
                "corner_indices": corner_index,
                "room_indices": room_index,
                "src_key_padding_mask": padding_mask,
                "connections": connections
            })
        
        return rooms, room_corner_bounds

    @staticmethod
    def _generate_structure_diffusion_feats(walls: typing.List[geometry.LineString]):
        """
        
        : param wall_arrays: list of arrays of shape (2, 2) where 2 is the number of corners in the wall
        """

        single_line_walls = turn_walls_into_single_lines(walls)

        wall_feats = []

        for wall in single_line_walls:
            assert wall.shape == (2, 2), f"Expected wall to have shape (2, 2), got {wall.shape}. That is a wall should have 2 points, each with 2 coordinates"

            point_1 = wall[0, :].reshape(1, 2)
            point_2 = wall[1, :].reshape(1, 2)

            # No connections array, because the walls are (2 pointed) lines, not polygons
            # Instead struct_corners_a should be used as the first point, and struct_corners_b as the second point

            wall_feats.append({
                "struct_corners_a": point_1,
                "struct_corners_b": point_2,
            })
        
        wall_dict = pd.DataFrame(wall_feats).to_dict(orient="list")
        wall_dict = {key: np.concatenate(value, axis=0) for key, value in wall_dict.items()}

        if not "struct_corners_a" in wall_dict:
            raise ValueError("No walls found in the structure")

        assert wall_dict["struct_corners_a"].shape[1] == 2, f"Expected struct_corners_a to have shape (N, 2), got {wall_dict['struct_corners_a'].shape}"

        return wall_dict

    def generate_house_diffusion_feats(self, process_subgraph_result: dict, strucural_lines: typing.List[geometry.LineString]):
        house = process_subgraph_result["house"]
        room_indices = process_subgraph_result["room_indices"]
        graph_edges = process_subgraph_result["graph_edges"]
        
        rooms, room_corner_bounds = self._generate_rooms_feats(house, room_indices)
        
        house_dict = pd.DataFrame(rooms).to_dict(orient="list")
        house_dict = {key: np.concatenate(value, axis=0) for key, value in house_dict.items()}

        number_of_points_in_layout = len(house_dict["corners"])

        # Set to number of points in layout as well, can be padded later
        max_num_points = len(house_dict["corners"])

        # Generate attention masks
        house_dict["self_mask"] = self.generate_self_mask(room_corner_bounds, max_num_points)
        house_dict["gen_mask"] = self.generate_gen_mask(number_of_points_in_layout, max_num_points)
        house_dict["door_mask"] = self.generate_door_mask(graph_edges, room_corner_bounds, max_num_points)

        house_dict["door_only_mask"] = self.generate_door_mask(graph_edges, room_corner_bounds, max_num_points, target_connectivity_types=["door"])
        house_dict["passage_only_mask"] = self.generate_door_mask(graph_edges, room_corner_bounds, max_num_points, target_connectivity_types=["passage"])
        house_dict["entrance_only_mask"] = self.generate_door_mask(graph_edges, room_corner_bounds, max_num_points, target_connectivity_types=["entrance"])

        stucture_dict = self._generate_structure_diffusion_feats(strucural_lines)

        number_of_walls = len(stucture_dict["struct_corners_a"])

        house_dict.update(stucture_dict)
        house_dict["struct_mask"] = self.generate_struct_mask(number_of_points_in_layout, number_of_walls)


        sample = HouseDiffusionSample(**house_dict)
        
        return sample




