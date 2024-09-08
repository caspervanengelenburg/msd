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
    
    id: str | int | None = None

    @property
    def number_of_corners(self):
        return len(self.corners)

    @property
    def number_of_walls(self):
        assert self.struct_corners_a.shape == self.struct_corners_b.shape

        assert self.struct_mask.shape[1] == self.struct_corners_a.shape[0]

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



    def get_feats(self, max_num_points: int, max_num_struct_walls: int, transpose_geometries=True, return_structural_feats=True):
        assert self.room_types.shape[0] <= max_num_points, f"Number of points {self.room_types.shape[0]} is greater than max number of points {max_num_points}"        

        door_mask_padded = self.pad_self_attn_mask(self.door_mask, max_num_points)
        self_mask_padded = self.pad_self_attn_mask(self.self_mask, max_num_points)
        gen_mask_padded = self.pad_self_attn_mask(self.gen_mask, max_num_points)

        corners = self.pad_features_matrix(self.corners, max_num_points)

        if transpose_geometries:
            corners = corners.transpose([1, 0])

        feats = {

            # Features that have to be tranformed when augmented geometrically
            "corners": corners,

            # Other features            
            "room_types": self.pad_features_matrix(self.room_types, max_num_points),
            "corner_indices": self.pad_features_matrix(self.corner_indices, max_num_points),
            "room_indices": self.pad_features_matrix(self.room_indices, max_num_points),
            
            # This feature should be of shape (N,) thus it should be squeezed
            # The padding mask should be inverted (0 if used, 1 if padding)
            "src_key_padding_mask": 1 - self.pad_features_matrix(self.src_key_padding_mask, max_num_points).squeeze(),
            
            "connections": self.pad_features_matrix(self.connections, max_num_points),

            # Masks 
            "door_mask": door_mask_padded,
            "self_mask": self_mask_padded,
            "gen_mask": gen_mask_padded,

            # Masks for the types of doors
            "door_only_mask": self.pad_self_attn_mask(self.door_only_mask, max_num_points),
            "passage_only_mask": self.pad_self_attn_mask(self.passage_only_mask, max_num_points),
            "entrance_only_mask": self.pad_self_attn_mask(self.entrance_only_mask, max_num_points),
        }

        if self.id is not None:
            feats["id"] = self.id

        if not return_structural_feats:
            return feats


        struct_mask_padded = self.pad_struct_attn_mask(self.struct_mask, max_num_points, max_num_struct_walls)

        struct_corners_a = self.pad_features_matrix(self.struct_corners_a, max_num_struct_walls)
        struct_corners_b = self.pad_features_matrix(self.struct_corners_b, max_num_struct_walls)

        if transpose_geometries:
            struct_corners_a = struct_corners_a.transpose([1, 0])
            struct_corners_b = struct_corners_b.transpose([1, 0])

        wall_self_mask = np.zeros((self.number_of_walls, self.number_of_walls))
        wall_self_mask_padded = self.pad_self_attn_mask(wall_self_mask, max_num_struct_walls)


        feats.update({
            "struct_corners_a": struct_corners_a,
            "struct_corners_b": struct_corners_b,

            "structural_mask": struct_mask_padded,

            "wall_self_mask": wall_self_mask_padded,
        })
        
        return feats
    
    def with_geometric_augmentation(self, augmentation_func: typing.Callable[[typing.List[np.ndarray]], typing.List[np.ndarray]]):
        """Returns a new HouseDiffusionSample with the geometric features augmented by `augmentation_fun`.
        
        :param augmentation: A function that takes in a list of features to augment, and returns the augmented features."""
        new_corners, new_struct_corners_a, new_struct_corners_b = augmentation_func([self.corners, self.struct_corners_a, self.struct_corners_b])

        return dataclasses.replace(self, corners=new_corners, struct_corners_a=new_struct_corners_a, struct_corners_b=new_struct_corners_b)

    def with_function_applied_to_geometry(self, f):
        return dataclasses.replace(self, corners=f(self.corners), struct_corners_a=f(self.struct_corners_a), struct_corners_b=f(self.struct_corners_b))

    def with_normalized_geometry(self, max_dim=512.0):

        # Map from [0, max_dim] to [-1, 1]
        return self.with_function_applied_to_geometry(lambda x: ((x / max_dim) - 0.5) * 2)

    def with_xy_corners_swapped(self):
        """Swaps the x and y coordinates of the corners."""

        # The corners were stored as y, x instead of x, y
        corners = self.corners[:, [1, 0]]

        return dataclasses.replace(self, corners=corners)

    # def with_tranposed_geometry(self):
    #     return self.with_function_applied_to_geometry(lambda x: x.transpose([1, 0]))


    def with_wall_i_masked_out(self, wall_to_mask_out):
        """Return a copy of the HouseDiffusionSample with the i-th wall masked out.
        
        This is accomplished by setting the corresponding column in the structural mask to 1."""
        
        assert self.struct_mask.shape[1] == self.number_of_walls, f"Structural mask should have shape (N, #walls), got {self.struct_mask.shape}"

        assert wall_to_mask_out < self.number_of_walls, f"Expected {wall_to_mask_out=} to be less than {self.number_of_walls=}"

        new_struct_mask = self.struct_mask.copy()

        new_struct_mask[:, wall_to_mask_out] = 1

        return dataclasses.replace(self, struct_mask=new_struct_mask)
    
    def with_randomized_room_indices(self):
        """Returns a copy of the HouseDiffusionSample with the room indices randomized.

        Only randomized up to the max used index.    
    
        """

        room_indices = self.room_indices.copy()
        
        rng = np.random.default_rng()

        max_index = np.max(np.where(room_indices == 1), axis=1)[1]

        ri_active = room_indices[:, :max_index + 1]

        room_indices[:, :max_index + 1] = rng.permutation(ri_active, axis=1)

        return dataclasses.replace(self, room_indices=room_indices)

    def with_randomized_wall_point_order(self):
        """Returns a copy of the HouseDiffusionSample with the wall point order randomized.

        Only randomized up to the max used index.    
        """

        assert self.struct_corners_a.shape[1] == 2

        struct_corners_a = self.struct_corners_a.copy()
        struct_corners_b = self.struct_corners_b.copy()

        rng = np.random.default_rng()

        max_index = struct_corners_a.shape[0]

        for i in range(max_index):
            if rng.random() > 0.5:

                # Should copy, otherwise a and b will become the same
                tmp_b, tmp_a = struct_corners_b[i, :].copy(), struct_corners_a[i, :].copy()

                struct_corners_a[i, :], struct_corners_b[i, :] = tmp_b, tmp_a
        
        return dataclasses.replace(self, struct_corners_a=struct_corners_a, struct_corners_b=struct_corners_b)