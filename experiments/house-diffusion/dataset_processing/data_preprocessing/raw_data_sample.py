

import os
import pickle
import numpy as np


class DataSample:
    def __init__(self, datapath, id, is_train=True, use_MRR=True) -> None:
        self.is_train = is_train

        self.paths = self.get_paths(datapath)

        self.id = id

        self.struct_in = self._load_npy_file("struct_in")
        
        if is_train:
            self.full_out = self._load_npy_file("full_out")

            self.out_img = self.full_out[:, :, 0].astype(np.uint8)

        self.graph_in = self._load_graph("graph_in")

        if not is_train:
            if use_MRR:
                # The notebook for creating test input graphs without #corner predictions outputs a folder named 'graph_pred'

                self.graph_pred = self._load_graph("graph_pred")
            else:

                # The notebook for creating test input graphs for all corners outputs a folder named 'graph_pred_n_corners'
                self.graph_pred = self._load_graph("graph_pred_n_corners")
        
        if is_train:
            self.graph_out = self._load_graph("graph_out")

        self.structural_img = self.struct_in[:, :, 0].astype(np.uint8)

        xys = self.struct_in[:, :, 1:]

        self.image_bounds = np.array([xys[0, 0], xys[-1, -1]])

        # # Check if bounds are swapped?
        # self.image_bounds[:, [0, 1]] = self.image_bounds[:, [1, 0]]

    @staticmethod
    def get_paths(datapath):
        
        path = {
            "full": datapath,
            "graph_in": os.path.join(datapath, 'graph_in'),
            "struct_in": os.path.join(datapath, 'struct_in'),
            "full_out": os.path.join(datapath, 'full_out'),
            "graph_out": os.path.join(datapath, 'graph_out'),
            "graph_pred": os.path.join(datapath, 'graph_pred'),
            "graph_pred_n_corners": os.path.join(datapath, 'graph_pred_n_corners'),
        }

        return path

    def _load_npy_file(self, type):
        stack = np.load(os.path.join(self.paths[type], f'{self.id}.npy'))

        return stack

    def _load_graph(self, type):
        return load_pickle(os.path.join(self.paths[type], f'{self.id}.pickle'))




def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
