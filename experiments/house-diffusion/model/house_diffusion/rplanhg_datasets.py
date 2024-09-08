import random
import typing
import torch as th

from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import os
import cv2 as cv
from tqdm import tqdm
from collections import defaultdict

from scipy import sparse

import gc

def load_rplanhg_structural_data(
    batch_size,
    analog_bit,
    target_set = 8,
    set_name = 'train',
) -> typing.Iterator[typing.Tuple[th.Tensor, typing.Dict[str, th.Tensor]]]:
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of target set {target_set}")
    deterministic = False if set_name=='train' else True    
    dataset = RPlanhgStructuralDataset(set_name, analog_bit, target_set, use_structural_elements=True)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader

SAVE_PATH = "processed_rplan_structural"

def make_non_manhattan(poly, polygon, house_poly):

    raise NotImplementedError

    # dist = abs(poly[2]-poly[0])
    # direction = np.argmin(dist)
    # center = poly.mean(0)
    # min = poly.min(0)
    # max = poly.max(0)

    # tmp = np.random.randint(3, 7)
    # new_min_y = center[1]-(max[1]-min[1])/tmp
    # new_max_y = center[1]+(max[1]-min[1])/tmp
    # if center[0]<128:
    #     new_min_x = min[0]-(max[0]-min[0])/np.random.randint(2,5)
    #     new_max_x = center[0]
    #     poly1=[[min[0], min[1]], [new_min_x, new_min_y], [new_min_x, new_max_y], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]]]
    # else:
    #     new_min_x = center[0]
    #     new_max_x = max[0]+(max[0]-min[0])/np.random.randint(2,5)
    #     poly1=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [new_max_x, new_max_y], [new_max_x, new_min_y], [max[0], min[1]]]

    # new_min_x = center[0]-(max[0]-min[0])/tmp
    # new_max_x = center[0]+(max[0]-min[0])/tmp
    # if center[1]<128:
    #     new_min_y = min[1]-(max[1]-min[1])/np.random.randint(2,5)
    #     new_max_y = center[1]
    #     poly2=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]], [new_max_x, new_min_y], [new_min_x, new_min_y]]
    # else:
    #     new_min_y = center[1]
    #     new_max_y = max[1]+(max[1]-min[1])/np.random.randint(2,5)
    #     poly2=[[min[0], min[1]], [min[0], max[1]], [new_min_x, new_max_y], [new_max_x, new_max_y], [max[0], max[1]], [max[0], min[1]]]
    # p1 = gm.Polygon(poly1)
    # iou1 = house_poly.intersection(p1).area/ p1.area
    # p2 = gm.Polygon(poly2)
    # iou2 = house_poly.intersection(p2).area/ p2.area
    # if iou1>0.9 and iou2>0.9:
    #     return poly
    # if iou1<iou2:
    #     return poly1
    # else:
    #     return poly2

get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]


def make_sparse(matrix_3d):
    return [sparse.bsr_array(sub) for sub in matrix_3d]

class RPlanhgStructuralDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False, use_structural_elements=True):
        """
        :param set_name: train or eval

        :param analog_bit: not sure what it means, but if False model seems more like the paper (e.g. interpolate points on wall AU augmentation)

        :param target_set: which floorplan to hold back for evaluation (those with target_set number of rooms will not be used for training?)

        :param use_structural_elements: whether to use structural elements (e.g. outline of house)
        """
        super().__init__()
        # base_dir = '../datasets/rplan'

        # Should be a zip archive of what normally would be in base_dir
        data_zip_path = '../datasets/rplan_json.zip'

        self.non_manhattan = non_manhattan
        self.set_name = set_name
        self.analog_bit = analog_bit
        self.target_set = target_set

        self.use_structural_elements = use_structural_elements
        
        self.subgraphs = []
        self.org_graph_edges = []
        self.org_houses = []

        # E.g. the outline of the house is a structural polygon
        self.org_structural_polygons: typing.List[typing.List[typing.Tuple[np.ndarray, int]]] = []

        max_num_points = 100
        max_num_room_corners = 31
        
        # Different to spot bugs
        max_num_structural_points = 75

        if os.path.exists(f'{SAVE_PATH}/rplan_{set_name}_{target_set}.npz'):
            data = np.load(f'{SAVE_PATH}/rplan_{set_name}_{target_set}.npz', allow_pickle=True)
            self.graphs = make_sparse(data['graphs'])
            self.houses = make_sparse(data['houses'])
            self.door_masks = make_sparse(data['door_masks'])
            self.self_masks = make_sparse(data['self_masks'])
            self.gen_masks = make_sparse(data['gen_masks'])
            self.num_coords = 2
            self.max_num_points = max_num_points

            self.max_num_structural_points = max_num_structural_points

            # Throw away unneeded columns
            structurals = data['structurals'][:, :, :2]

            self.structurals = structurals
            self.struct_masks = make_sparse(data['struct_masks'])

            self.file_names = data['file_names']

            self.ids = [int(name.split('/')[1].split('.')[0]) for name in self.file_names]

            # cnumber_dist = np.load(f'{SAVE_PATH}/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()

            # Not need unless want to eval with synthetic room types?
            # if self.set_name == 'eval':
            #     data = np.load(f'{SAVE_PATH}/rplan_{set_name}_{target_set}_syn.npz', allow_pickle=True)
            #     self.syn_graphs = data['graphs']
            #     self.syn_houses = data['houses']
            #     self.syn_door_masks = data['door_masks']
            #     self.syn_self_masks = data['self_masks']
            #     self.syn_gen_masks = data['gen_masks']

            #     self.syn_structurals = structurals
            #     self.syn_struct_masks = data['struct_masks']

            #     self.syn_file_names = data['file_names']
            
            gc.collect()

        else:
            if self.set_name == 'eval':
                cnumber_dist = np.load(f'{SAVE_PATH}/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()


            os.makedirs(SAVE_PATH, exist_ok=True)

            # with open(f'{base_dir}/list.txt') as f:
            #     lines = f.readlines()

            import zipfile

            data_zip = zipfile.ZipFile(data_zip_path, 'r')

            # Get file names from zip archive that end with .json
            file_names = [f for f in data_zip.namelist() if f.endswith('.json')]

            cnt=0

            potential_subgraphs = []

            for file_name in tqdm(file_names):
                cnt=cnt+1
                
                with data_zip.open(file_name) as f:
                    try:
                        rms_type, rms_bbs, fp_eds, eds_to_rms = reader(f)
                    except json.JSONDecodeError as e:
                        print(f'Error reading {file_name}: {e}')
                        raise e
                
                fp_size = len([x for x in rms_type if x != 15 and x != 17])
                if self.set_name=='train' and fp_size == target_set:
                        continue
                if self.set_name=='eval' and fp_size != target_set:
                        continue
                a = [rms_type, rms_bbs, fp_eds, eds_to_rms]


                potential_subgraphs.append((a, file_name))
            
            orig_file_names = []

            for subgraph, file_name in tqdm(potential_subgraphs):

                try:
                    graph_edges, house, structural_polygons = self.process_subgraph(subgraph, self.set_name)
                    self.org_graph_edges.append(graph_edges)
                    self.org_houses.append(house)
                    self.org_structural_polygons.append(structural_polygons)

                    self.subgraphs.append(graph_edges)

                    orig_file_names.append(file_name)
                    
                except:
                    pass

            houses = []
            door_masks = []
            self_masks = []
            gen_masks = []
            
            structurals = []
            struct_masks = []

            file_names = []

            graphs = []
            if self.set_name=='train':
                cnumber_dist = defaultdict(list)

            if self.non_manhattan:
                print("not going to make changes to dataset...")
                raise NotImplementedError
                # for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                #     # Generating non-manhattan Balconies
                #     tmp = []
                #     for i, room in enumerate(h):
                #         if room[1]>10:
                #             continue
                #         if len(room[0])!=4: 
                #             continue
                #         if np.random.randint(2):
                #             continue
                #         poly = gm.Polygon(room[0])
                #         house_polygon = unary_union([gm.Polygon(room[0]) for room in h])
                #         room[0] = make_non_manhattan(room[0], poly, house_polygon)

            for h, graph_edges, structural_polygons, file_name in tqdm(zip(self.org_houses, self.org_graph_edges, self.org_structural_polygons, orig_file_names), desc='processing dataset'):
                house = []
                corner_bounds = []
                num_points = 0
                
                try:
                    for i, room in enumerate(h):
                        
                        # Relabeling room types
                        if room[1]>10:
                            room[1] = {15:11, 17:12, 16:13}[room[1]]
                        
                        room[0] = np.reshape(room[0], [len(room[0]), 2])/256. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                        room[0] = room[0] * 2 # map to [-1, 1]
                        
                        # Adding conditions
                        num_room_corners = len(room[0])

                        if num_room_corners > max_num_room_corners:
                            raise ValueError(f"Expected room corners to fit, but got {num_room_corners} > {max_num_room_corners}")

                        # Store how many corners this room type has
                        if self.set_name=='train':
                            cnumber_dist[room[1]].append(len(room[0]))
                        
                        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
                        room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                        # Src_key_padding_mask
                        padding_mask = np.repeat(1, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                        # Generating corner bounds for attention masks
                        connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                        connections += num_points
                        corner_bounds.append([num_points, num_points+num_room_corners])
                        num_points += num_room_corners
                        room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections), 1)
                        house.append(room)
                except ValueError:
                    continue

                house_layouts = np.concatenate(house, 0)
                
                if len(house_layouts)>max_num_points:
                    continue
            
                padding = np.zeros((max_num_points-len(house_layouts), 94))
                house_layouts_padded = np.concatenate((house_layouts, padding), 0)

                # Attention mask for Global Self Attention in the paper?
                gen_mask = np.ones((max_num_points, max_num_points))
                gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                
                structural_layouts = self._extract_structural_layouts(max_num_structural_points, structural_polygons)

                # Attention mask for cross attention between corners of rooms and corners of structural elements
                struct_mask = np.ones((max_num_points, max_num_structural_points))
                struct_mask[:len(house_layouts), :len(structural_layouts)] = 0

                structural_padding = np.zeros((max_num_structural_points-len(structural_layouts), 94))
                structural_layouts_padded = np.concatenate((structural_layouts, structural_padding), 0)

                
                # Attention mask for Relational Cross Attention in the paper?
                door_mask = np.ones((max_num_points, max_num_points))

                # Attention mask for Component-wise Self Attention in the paper?
                self_mask = np.ones((max_num_points, max_num_points))
                
                for i in range(len(corner_bounds)):
                    for j in range(len(corner_bounds)):
                        if i==j:
                            self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                        elif any(np.equal([i, 1, j], graph_edges).all(1)) or any(np.equal([j, 1, i], graph_edges).all(1)):
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0

                houses.append(house_layouts_padded)
                door_masks.append(door_mask)
                self_masks.append(self_mask)
                gen_masks.append(gen_mask)

                structurals.append(structural_layouts_padded)
                struct_masks.append(struct_mask)

                graphs.append(graph_edges)

                file_names.append(file_name)

            self.max_num_points = max_num_points
            self.max_num_structural_points = max_num_structural_points
            self.houses = houses
            self.door_masks = door_masks
            self.self_masks = self_masks
            self.gen_masks = gen_masks
            self.num_coords = 2
            self.graphs = graphs

            self.structurals = structurals
            self.struct_masks = struct_masks

            self.file_names = file_names

            np.savez_compressed(f'{SAVE_PATH}/rplan_{set_name}_{target_set}', graphs=self.graphs, houses=self.houses,
                    door_masks=self.door_masks, self_masks=self.self_masks, gen_masks=self.gen_masks, 
                    structurals=self.structurals, struct_masks=self.struct_masks,
                    file_names=self.file_names)
            
            if self.set_name=='train':
                np.savez_compressed(f'{SAVE_PATH}/rplan_{set_name}_{target_set}_cndist', cnumber_dist=cnumber_dist)

            if set_name=='eval':
                houses = []
                graphs = []
                door_masks = []
                self_masks = []
                gen_masks = []
                len_house_layouts = 0

                structurals = []
                struct_masks = []

                file_names = []

                for h, graph_edges, structural_polygons, file_name in tqdm(zip(self.org_houses, self.org_graph_edges, self.org_structural_polygons, orig_file_names), desc='processing dataset'):
                    house = []
                    corner_bounds = []
                    num_points = 0
                    num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    while np.sum(num_room_corners_total)>=max_num_points:
                        num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    for i, room in enumerate(h):
                        # Adding conditions
                        num_room_corners = num_room_corners_total[i]
                        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
                        room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                        # Src_key_padding_mask
                        padding_mask = np.repeat(1, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                        # Generating corner bounds for attention masks
                        connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                        connections += num_points
                        corner_bounds.append([num_points, num_points+num_room_corners])
                        num_points += num_room_corners
                        room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index, padding_mask, connections), 1)
                        house.append(room)

                    house_layouts = np.concatenate(house, 0)
                    if np.sum([len(room[0]) for room in h])>max_num_points:
                        continue
                    padding = np.zeros((max_num_points-len(house_layouts), 94))
                    
                    gen_mask = np.ones((max_num_points, max_num_points))
                    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                    

                    structural_layouts = self._extract_structural_layouts(max_num_structural_points, structural_polygons)

                    # Attention mask for cross attention between corners of rooms and corners of structural elements
                    struct_mask = np.ones((max_num_points, max_num_structural_points))
                    struct_mask[:len(house_layouts), :len(structural_layouts)] = 0

                    structural_padding = np.zeros((max_num_structural_points-len(structural_layouts), 94))
                    structural_layouts_padded = np.concatenate((structural_layouts, structural_padding), 0)

                    house_layouts_padded = np.concatenate((house_layouts, padding), 0)

                    door_mask = np.ones((max_num_points, max_num_points))
                    self_mask = np.ones((max_num_points, max_num_points))
                    for i, room in enumerate(h):
                        if room[1]==1:
                            living_room_index = i
                            break
                    for i in range(len(corner_bounds)):
                        is_connected = False
                        for j in range(len(corner_bounds)):
                            if i==j:
                                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                            elif any(np.equal([i, 1, j], graph_edges).all(1)) or any(np.equal([j, 1, i], graph_edges).all(1)):
                                door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                                is_connected = True
                        if not is_connected:
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

                    houses.append(house_layouts_padded)
                    door_masks.append(door_mask)
                    self_masks.append(self_mask)
                    gen_masks.append(gen_mask)
                    graphs.append(graph_edges)

                    structurals.append(structural_layouts_padded)
                    struct_masks.append(struct_mask)

                    file_names.append(file_name)

                self.syn_houses = houses
                self.syn_door_masks = door_masks
                self.syn_self_masks = self_masks
                self.syn_gen_masks = gen_masks
                self.syn_graphs = graphs

                self.syn_structurals = structurals
                self.syn_struct_masks = struct_masks

                self.syn_file_names = file_names

                np.savez_compressed(f'{SAVE_PATH}/rplan_{set_name}_{target_set}_syn', graphs=self.syn_graphs, houses=self.syn_houses,
                        door_masks=self.syn_door_masks, self_masks=self.syn_self_masks, gen_masks=self.syn_gen_masks, 
                        structurals=self.syn_structurals, struct_masks=self.syn_struct_masks,
                        file_names=self.syn_file_names)

    @staticmethod
    def _extract_structural_layouts(max_num_structural_points, structural_polygons):
        structural_elements = []
        structural_corner_bounds = []
        sturctural_num_points = 0

        # Process structural outlines like rooms
        for i, structural_element in enumerate(structural_polygons):
            outline, struct_type = structural_element
                    
            outline = np.reshape(outline, [len(outline), 2])/256. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
            outline = outline * 2 # map to [-1, 1]

                    # TODO: keep track of statistics how many corners structural elements have
                                    
                    # Adding conditions
            num_structural_element_corners = len(outline)

            rtype = np.repeat(np.array([get_one_hot(struct_type, 25)]), num_structural_element_corners, 0)
                    
            element_index = np.repeat(np.array([get_one_hot(len(structural_elements)+1, 8)]), num_structural_element_corners, 0)
            corner_index = np.array([get_one_hot(x, 56) for x in range(num_structural_element_corners)])
                    
                    # Src_key_padding_mask
            padding_mask = np.repeat(1, num_structural_element_corners)
            padding_mask = np.expand_dims(padding_mask, 1)
                    
                    # Generating corner bounds for attention masks
            connections = np.array([[i,(i+1) % num_structural_element_corners] for i in range(num_structural_element_corners)])
            connections += sturctural_num_points
                    
            structural_corner_bounds.append([sturctural_num_points, sturctural_num_points+num_structural_element_corners])
            sturctural_num_points += num_structural_element_corners
            structural_element = np.concatenate((outline, rtype, corner_index, element_index, padding_mask, connections), 1)
                    
            structural_elements.append(structural_element)
                
        structural_layouts = np.concatenate(structural_elements, 0)

        if len(structural_layouts) > max_num_structural_points:
            raise RuntimeError(f"Expected stuctural layouts to fit, but got {len(structural_layouts)} > {max_num_structural_points}")
        
        return structural_layouts

    @staticmethod
    def process_subgraph(graph: typing.Tuple, split_set_name: str) -> typing.Tuple[np.ndarray, typing.List[np.ndarray], typing.List[typing.Tuple[np.ndarray, int]]]:
        """Process subgraph
        
        :param graph: tuple of rms_type, rms_bbs, fp_eds, eds_to_rms
        :param split_set_name: the name of the split that is being processed

        :return: graph_edges, house, interior_contour
        """

        rms_type, rms_bbs, fp_eds, eds_to_rms = graph

        rms_bbs = np.array(rms_bbs)
        fp_eds = np.array(fp_eds)

                # extract boundary box and centralize
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl+br)/2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        tl -= shift
        br -= shift

                # build input graph
        graph_nodes, graph_edges, rooms_mks = RPlanhgStructuralDataset.build_graph(split_set_name, rms_type, fp_eds, eds_to_rms)

        house = []
        for room_mask, room_type in zip(rooms_mks, graph_nodes):
            room_mask = room_mask.astype(np.uint8)
            room_mask = cv.resize(room_mask, (256, 256), interpolation = cv.INTER_AREA)
            contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours = contours[0]
            house.append([contours[:,0,:], room_type])

        # Compute interior contour
        interior = np.sum(rooms_mks[(graph_nodes != 15) & (graph_nodes != 17)], axis=0)

        interior_mask = interior.astype(np.uint8)
        interior_mask = cv.resize(interior_mask, (256, 256), interpolation = cv.INTER_AREA)
        interior_contour, _ = cv.findContours(interior_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        interior_contour: np.ndarray = interior_contour[0][:, 0, :]

        INTERIOR_TYPE = 1

        # Currently only the interior contour is used as structural element, but it could in theory be more elements
        structural_polygon = (interior_contour, INTERIOR_TYPE)

        return graph_edges, house, [structural_polygon]

    def __len__(self):
        return len(self.houses)

    @staticmethod
    def pad_features_matrix(features_matrix: np.ndarray, max_num_points: int):
        assert features_matrix.shape[0] < max_num_points, f"Number of points {features_matrix.shape[0]} is greater than max number of points {max_num_points}"

        assert len(features_matrix.shape) == 2, f"Expected features_matrix to have shape (N, F), got {features_matrix.shape}"

        feats_dim = features_matrix.shape[1]

        padding = np.zeros((max_num_points - features_matrix.shape[0], feats_dim))

        features_matrix_padded = np.concatenate((features_matrix, padding), axis=0)

        return features_matrix_padded

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
    # Method to convert structure polygon (outer walls are a polygonal line)
    def make_struct_corners_a_b(structural_arr, struct_mask):
        # struct_mask = struct_masks[0]
        # struct = structurals[0][:, :2]

        # The maximum number of walls/line segments, is the same as the maximum number of corners of the outer walls polygon
        max_num_walls = structural_arr.shape[0]

        # Select the not masked out structure points part of the polygon
        selection_struct_points = (struct_mask.min(axis=0) == 0)

        valid_struct = structural_arr[selection_struct_points]

        # Now should create two arrays, struct_corners_a, and struct_corners_b, where a is the starting point and b end points of all linesegments

        # the start points:
        struct_corners_a = valid_struct.copy()

        # the end points:

        # shift by -1 to put each element at the previous position (rolling around at the end, so first becomes last). 
        # Thus, e.g. the second point, which is the end point of the first line segment (which has start point at struct_corners_a[0, :]), 
        # is now at struct_corners_b[0, :], which is correct.
        struct_corners_b = np.roll(valid_struct, shift=-1, axis=0)

        # Zero pad to original length
        struct_corners_a = RPlanhgStructuralDataset.pad_features_matrix(struct_corners_a, max_num_walls)
        struct_corners_b = RPlanhgStructuralDataset.pad_features_matrix(struct_corners_b, max_num_walls)

        return struct_corners_a, struct_corners_b

    @staticmethod
    def make_wall_self_mask(struct_mask):

        # The maximum number of walls/line segments
        max_num_struct_walls = struct_mask.shape[1]

        # Select the not masked out structure points part of the polygon
        selection_struct_points = (struct_mask.min(axis=0) == 0)

        # Count how many walls there are
        number_of_walls = selection_struct_points.sum()

        wall_self_mask = np.zeros((number_of_walls, number_of_walls))
        wall_self_mask_padded = RPlanhgStructuralDataset.pad_self_attn_mask(wall_self_mask, max_num_struct_walls)

        return wall_self_mask_padded


    def __getitem__(self, idx):
        # idx = int(idx//20)

        house = self.houses[idx].toarray()

        arr = house[:, :self.num_coords]
        structural_arr = self.structurals[idx][:, :self.num_coords]

        g = self.graphs[idx].toarray()

        graph = np.concatenate((g, np.zeros([200-len(g), 3])), 0)

        struct_mask = self.struct_masks[idx].toarray()

        door_mask = self.door_masks[idx].toarray()

        # Should be completely masked out, because all rooms are connected by doors, and none are by passages or entrances
        passage_only_mask = np.ones_like(door_mask)
        entrance_only_mask = np.ones_like(door_mask)

        cond = {
                # 'door_mask' was replaced with specific masks per connection type: door_only_mask, passage_only_mask, entrance_only_mask
                # 'door_mask': self.door_masks[idx],
            
                'door_only_mask': door_mask,
                'passage_only_mask': passage_only_mask,
                'entrance_only_mask': entrance_only_mask,

                'self_mask': self.self_masks[idx].toarray(),
                'gen_mask': self.gen_masks[idx].toarray(),
                'room_types': house[:, self.num_coords:self.num_coords+25],
                'corner_indices': house[:, self.num_coords+25:self.num_coords+57],
                'room_indices': house[:, self.num_coords+57:self.num_coords+89],
                'src_key_padding_mask': 1-house[:, self.num_coords+89],
                'connections': house[:, self.num_coords+90:self.num_coords+92],

                'structural_mask': struct_mask,

                'wall_self_mask': self.make_wall_self_mask(struct_mask),
                
                # not needed now that model uses struct_corners_a and struct_corners_b
                # 'structural_connections': self.structurals[idx][:, -2:],

                'graph': graph, # Only used for evaluation
                
                # id based on rplan file name
                'id': self.ids[idx]
                }
        
        # Only needed if want to evaluate with synthetic room graphs I think
        # if self.set_name == 'eval':
        #     syn_graph = np.concatenate((self.syn_graphs[idx], np.zeros([200-len(self.syn_graphs[idx]), 3])), 0)
        #     assert (graph == syn_graph).all(), idx
        #     cond.update({
        #         'syn_door_mask': self.syn_door_masks[idx],
        #         'syn_self_mask': self.syn_self_masks[idx],
        #         'syn_gen_mask': self.syn_gen_masks[idx],
        #         'syn_room_types': self.syn_houses[idx][:, self.num_coords:self.num_coords+25],
        #         'syn_corner_indices': self.syn_houses[idx][:, self.num_coords+25:self.num_coords+57],
        #         'syn_room_indices': self.syn_houses[idx][:, self.num_coords+57:self.num_coords+89],
        #         'syn_src_key_padding_mask': 1-self.syn_houses[idx][:, self.num_coords+89],
        #         'syn_connections': self.syn_houses[idx][:, self.num_coords+90:self.num_coords+92],
        #         'syn_graph': syn_graph, # Only used for evaluation

        #         # probably not used... maybe for eval
        #         # 'syn_structural_mask': self.syn_struct_masks[idx],
        #         # 'syn_structural_connections': self.syn_structurals[idx][:, -2:],
        #         })
            
        if self.set_name == 'train':
            rotation = random.randint(0,3)

            arr = rotate(arr, rotation)
            structural_arr = rotate(structural_arr, rotation)

        if not self.analog_bit:
            arr = np.transpose(arr, [1, 0])

            struct_corners_a, struct_corners_b = self.make_struct_corners_a_b(structural_arr, struct_mask)

            # structural_arr = np.transpose(structural_arr, [1, 0])

            struct_corners_a = np.transpose(struct_corners_a, [1, 0])
            struct_corners_b = np.transpose(struct_corners_b, [1, 0])


            cond.update({
                # Not used anymore, use struct_corners_a, and struct_corners_b instead
                # 'x_structural': structural_arr,

                "struct_corners_a": struct_corners_a,
                "struct_corners_b": struct_corners_b,

            })
            
            return arr.astype(float), cond
        else:
            raise NotImplementedError("Expected analog_bit to be False")
            # ONE_HOT_RES = 256
            # arr_onehot = np.zeros((ONE_HOT_RES*2, arr.shape[1])) - 1
            # xs = ((arr[:, 0]+1)*(ONE_HOT_RES/2)).astype(int)
            # ys = ((arr[:, 1]+1)*(ONE_HOT_RES/2)).astype(int)
            # xs = np.array([get_bin(x, 8) for x in xs])
            # ys = np.array([get_bin(x, 8) for x in ys])
            # arr_onehot = np.concatenate([xs, ys], 1)
            # arr_onehot = np.transpose(arr_onehot, [1, 0])
            # arr_onehot[arr_onehot==0] = -1
            # return arr_onehot.astype(float), cond

    @staticmethod
    def make_sequence(edges):
        polys = []
        v_curr = tuple(edges[0][:2])
        e_ind_curr = 0
        e_visited = [0]
        seq_tracker = [v_curr]
        find_next = False
        while len(e_visited) < len(edges):
            if find_next == False:
                if v_curr == tuple(edges[e_ind_curr][2:]):
                    v_curr = tuple(edges[e_ind_curr][:2])
                else:
                    v_curr = tuple(edges[e_ind_curr][2:])
                find_next = not find_next 
            else:
                # look for next edge
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        if (v_curr == tuple(e[:2])):
                            v_curr = tuple(e[2:])
                            e_ind_curr = k
                            e_visited.append(k)
                            break
                        elif (v_curr == tuple(e[2:])):
                            v_curr = tuple(e[:2])
                            e_ind_curr = k
                            e_visited.append(k)
                            break

            # extract next sequence
            if v_curr == seq_tracker[-1]:
                polys.append(seq_tracker)
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        v_curr = tuple(edges[0][:2])
                        seq_tracker = [v_curr]
                        find_next = False
                        e_ind_curr = k
                        e_visited.append(k)
                        break
            else:
                seq_tracker.append(v_curr)
        polys.append(seq_tracker)

        return polys

    @staticmethod
    def build_graph(split_set_name, rms_type, fp_eds, eds_to_rms, out_size=64):
        # create edges
        triples = []
        nodes = rms_type 
        # encode connections
        for k in range(len(nodes)):
            for l in range(len(nodes)):
                if l > k:
                    is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                    if is_adjacent:
                        if 'train' in split_set_name:
                            triples.append([k, 1, l])
                        else:
                            triples.append([k, 1, l])
                    else:
                        if 'train' in split_set_name:
                            triples.append([k, -1, l])
                        else:
                            triples.append([k, -1, l])
        # get rooms masks
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):                  
            eds_to_rms_tmp.append([eds_to_rms[l][0]])
        rms_masks = []
        im_size = 256
        fp_mk = np.zeros((out_size, out_size))
        for k in range(len(nodes)):
            # add rooms and doors
            eds = []
            for l, e_map in enumerate(eds_to_rms_tmp):
                if (k in e_map):
                    eds.append(l)
            # draw rooms
            rm_im = Image.new('L', (im_size, im_size))
            dr = ImageDraw.Draw(rm_im)
            for eds_poly in [eds]:
                poly = RPlanhgStructuralDataset.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
                poly = [(im_size*x, im_size*y) for x, y in poly]
                if len(poly) >= 2:
                    dr.polygon(poly, fill='white')
                else:
                    print("Empty room")
                    exit(0)
            rm_im = rm_im.resize((out_size, out_size))
            rm_arr = np.array(rm_im)
            inds = np.where(rm_arr>0)
            rm_arr[inds] = 1.0
            rms_masks.append(rm_arr)
            if rms_type[k] != 15 and rms_type[k] != 17:
                fp_mk[inds] = k+1
        # trick to remove overlap
        for k in range(len(nodes)):
            if rms_type[k] != 15 and rms_type[k] != 17:
                rm_arr = np.zeros((out_size, out_size))
                inds = np.where(fp_mk==k+1)
                rm_arr[inds] = 1.0
                rms_masks[k] = rm_arr
        # convert to array
        nodes = np.array(nodes)
        triples = np.array(triples)
        rms_masks = np.array(rms_masks)
        return nodes, triples, rms_masks

def is_adjacent(box_a, box_b, threshold=0.03):
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b
    h1, h2 = x1-x0, x3-x2
    w1, w2 = y1-y0, y3-y2
    xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
    yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0
    delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
    delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0
    delta = max(delta_x, delta_y)
    return delta < threshold

def reader(f: typing.IO[bytes]):
    info =json.load(f)

    # room bounding boxes
    rms_bbs=np.asarray(info['boxes'])

    # floorplan edges
    fp_eds=info['edges']

    # rooms type
    rms_type=info['room_type']

    # edges to rooms
    eds_to_rms=info['ed_rm']

    s_r=0
    for rmk in range(len(rms_type)):
        if(rms_type[rmk]!=17):
            s_r=s_r+1  

    # normalize room bounding boxes (coordinates from 0 to 256? to 0:1)
    rms_bbs = np.array(rms_bbs)/256.0

    # normalized room edges
    fp_eds = np.array(fp_eds)/256.0

    fp_eds = fp_eds[:, :4]

    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    
    shift = (tl+br)/2.0 - 0.5
    
    rms_bbs[:, :2] -= shift 
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift 
    tl -= shift
    br -= shift
    
    return rms_type, rms_bbs, fp_eds, eds_to_rms

def rotate(arr, rotation):
    if rotation == 1:
        arr[:, [0, 1]] = arr[:, [1, 0]]
        arr[:, 0] = -arr[:, 0]
    elif rotation == 2:
        arr[:, [0, 1]] = -arr[:, [1, 0]]
    elif rotation == 3:
        arr[:, [0, 1]] = arr[:, [1, 0]]
        arr[:, 1] = -arr[:, 1]
    
    return arr

# def random_augment_rotate(arr):
#     #### Random Rotate
#     rotation = random.randint(0,3)

#     arr = rotate(arr, rotation)
    

#     ## To generate any rotation uncomment this

#     # if self.non_manhattan:
#         # theta = random.random()*np.pi/2
#         # rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
#                     # [np.sin(theta), np.cos(theta), 0]])
#         # arr = np.matmul(arr,rot_mat)[:,:2]

#     # Random Scale
#     # arr = arr * np.random.normal(1., .5)

#     # Random Shift
#     # arr[:, 0] = arr[:, 0] + np.random.normal(0., .1)
#     # arr[:, 1] = arr[:, 1] + np.random.normal(0., .1)

#     return arr

if __name__ == '__main__':
    dataset = RPlanhgStructuralDataset('eval', False, 8)
