import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class HouseDiffusionSample:

    corners: np.ndarray

    door_mask: np.ndarray
    self_mask: np.ndarray
    gen_mask: np.ndarray

    door_only_mask: np.ndarray
    passage_only_mask: np.ndarray
    entrance_only_mask: np.ndarray

    # Features of #corners x #features
    room_types: np.ndarray
    corner_indices: np.ndarray
    room_indices: np.ndarray
    
    # 1 if the point is used, 0 if it is padding
    src_key_padding_mask: np.ndarray

    connections: np.ndarray

    struct_corners_a: np.ndarray
    struct_corners_b: np.ndarray

    struct_mask: np.ndarray

    @property
    def number_of_corners(self):
        return len(self.corners)

    @property
    def number_of_walls(self):
        assert self.struct_corners_a.shape == self.struct_corners_b.shape

        return len(self.struct_corners_a)

    @staticmethod
    def pad_self_attn_mask(attn_mask: np.ndarray, max_num_points: int):
        assert attn_mask.shape[0] == attn_mask.shape[1], "Attention mask should be square"
        assert attn_mask.shape[0] < max_num_points, f"Size of attn_mask {attn_mask.shape[0]} should have been less than {max_num_points}"

        # Padding should be ones, because the padding should be masked out
        attn_mask_padded = np.ones((max_num_points, max_num_points))

        # Set the used part of the mask
        attn_mask_padded[:attn_mask.shape[0], :attn_mask.shape[1]] = attn_mask

        return attn_mask_padded


    @staticmethod
    def pad_features_matrix(features_matrix: np.ndarray, max_num_points: int):
        assert features_matrix.shape[0] < max_num_points, f"Number of points {features_matrix.shape[0]} is greater than max number of points {max_num_points}"

        assert len(features_matrix.shape) == 2, f"Expected features_matrix to have shape (N, F), got {features_matrix.shape}"

        feats_dim = features_matrix.shape[1]

        padding = np.zeros((max_num_points - features_matrix.shape[0], feats_dim))

        features_matrix_padded = np.concatenate((features_matrix, padding), axis=0)

        return features_matrix_padded
    
    @staticmethod
    def pad_struct_attn_mask(struct_attn_mask: np.ndarray, max_num_points: int, max_num_walls: int):
        assert struct_attn_mask.shape[0] <= max_num_points
        assert struct_attn_mask.shape[1] <= max_num_walls

        # Padding should be ones, because the padding should be masked out
        struct_attn_mask_padded = np.ones((max_num_points, max_num_walls))

        # Set the used part of the mask
        struct_attn_mask_padded[:struct_attn_mask.shape[0], :struct_attn_mask.shape[1]] = struct_attn_mask

        return struct_attn_mask_padded



    def get_feats(self, max_num_points: int, max_num_struct_walls: int, transpose_geometries=True):
        assert self.room_types.shape[0] <= max_num_points, f"Number of points {self.room_types.shape[0]} is greater than max number of points {max_num_points}"        

        door_mask_padded = self.pad_self_attn_mask(self.door_mask, max_num_points)
        self_mask_padded = self.pad_self_attn_mask(self.self_mask, max_num_points)
        gen_mask_padded = self.pad_self_attn_mask(self.gen_mask, max_num_points)

        struct_mask_padded = self.pad_struct_attn_mask(self.struct_mask, max_num_points, max_num_struct_walls)

        corners = self.pad_features_matrix(self.corners, max_num_points)

        struct_corners_a = self.pad_features_matrix(self.struct_corners_a, max_num_struct_walls)
        struct_corners_b = self.pad_features_matrix(self.struct_corners_b, max_num_struct_walls)

        if transpose_geometries:
            corners = corners.transpose([1, 0])
            struct_corners_a = struct_corners_a.transpose([1, 0])
            struct_corners_b = struct_corners_b.transpose([1, 0])

        return {

            # Features that have to be tranformed when augmented geometrically
            "corners": corners,

            "struct_corners_a": struct_corners_a,
            "struct_corners_b": struct_corners_b,

            # Other features            
            "room_types": self.pad_features_matrix(self.room_types, max_num_points),
            "corner_indices": self.pad_features_matrix(self.corner_indices, max_num_points),
            "room_indices": self.pad_features_matrix(self.room_indices, max_num_points),
            
            # This feature should be of shape (N,) thus it should be squeezed
            "src_key_padding_mask": self.pad_features_matrix(self.src_key_padding_mask, max_num_points).squeeze(),
            
            "connections": self.pad_features_matrix(self.connections, max_num_points),

            # Masks 
            "door_mask": door_mask_padded,
            "self_mask": self_mask_padded,
            "gen_mask": gen_mask_padded,

            "structural_mask": struct_mask_padded
        }
    
    def with_geometric_augmentation(self, augmentation_func: typing.Callable[[typing.List[np.ndarray]], typing.List[np.ndarray]]):
        """Returns a new HouseDiffusionSample with the geometric features augmented by `augmentation_fun`.
        
        :param augmentation: A function that takes in a list of features to augment, and returns the augmented features."""
        new_corners, new_struct_corners_a, new_struct_corners_b = augmentation_func([self.corners, self.struct_corners_a, self.struct_corners_b])

        return dataclasses.replace(self, corners=new_corners, struct_corners_a=new_struct_corners_a, struct_corners_b=new_struct_corners_b)

    def with_function_applied_to_geometry(self, f):
        return dataclasses.replace(self, corners=f(self.corners), struct_corners_a=f(self.struct_corners_a), struct_corners_b=f(self.struct_corners_b))

    def with_normalized_geometry(self, max_dim=512.0):
        return self.with_function_applied_to_geometry(lambda x: (x / max_dim) - 0.5)

    # def with_tranposed_geometry(self):
    #     return self.with_function_applied_to_geometry(lambda x: x.transpose([1, 0]))
