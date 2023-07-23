import numpy as np
import tensorflow as tf
from copy import deepcopy


class DataInfo:
    """
    Information of dateset or buffer.
    """

    def __init__(self, names: tuple or list, shapes: tuple or list, dtypes, dataset=None):
        """Data info struct.

        Args:
            names (tuple): data names.
            shapes (tuple): shapes 
            dtypes (tuple, optional): dtypes.
        """

        # TODO: 当前buffer size
        self.shape_dict = dict(zip(names, shapes))

        self.names = names

        if isinstance(dtypes, tuple) or isinstance(dtypes, list):
            self.type_dict = dict(zip(names, dtypes))
        else:
            self.type_dict = dict()
            for key in names:
                self.type_dict[key] = dtypes
        if dataset is not None:
            self.dataset_dict = dict(zip(names, dataset))
        else:
            self.dataset_dict = None

    def add_info(self, name: str, shape, dtype):
        """add info.

        Args:
            name (str): name.
            shape (tuple): shape.
            dtype (type): dtype.
        """
        self.shape_dict[name] = shape
        self.type_dict[name] = dtype

        # add element to names
        names = list(self.names)
        names.append(name)
        self.names = tuple(names)

    @property
    def get_total_size(self):
        """get batch size.
        """
        return self.shape_dict[self.names[0]][0]

    def keys(self):
        return self.names


class DataSet:

    def __init__(self,
                 data_dict: dict,
                 ):
        """
        Args:
            data_dict (dict): data dict.
        """
        self.data_dict = data_dict
        names = tuple(self.data_dict.keys())
        # self.batch_size = 32

        self.buffer_size = self.data_dict[names[0]].shape[0]

    def slice_data(self, start: int, end: int):
        """slice data.

        Args:
            start (int): start index.
            end (int): end index.
        """
        new_data_dict = {}
        for key in self.data_dict.keys():
            new_data_dict[key] = self.data_dict[key][start:end]

        return DataSet(new_data_dict)

    def __call__(self, batch_size: int):
        """
        sample data
        """

        indices = np.random.permutation(self.buffer_size)

        start_index = 0

        while start_index < self.buffer_size:
            end_index = min(start_index + batch_size, self.buffer_size)
            batch_indices = indices[start_index:end_index]

            batch = {}
            for key, val in self.data_dict.items():
                batch[key] = tf.cast(deepcopy(val[batch_indices]), tf.float32)

            yield batch

            start_index = end_index
