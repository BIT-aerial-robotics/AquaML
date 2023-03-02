import abc
from AquaML.BaseClass import BaseAlgo
from AquaML.data.DataPool import DataPool
from AquaML.data.ArgsPool import ArgsPool
import os
import json


# import time

def mkdir(path: str):
    """
    create a directory in current path.

    Args:
        path (_type_:str): name of directory.

    Returns:
        _type_: str or None: path of directory.
    """
    current_path = os.getcwd()
    # print(current_path)
    path = os.path.join(current_path, path)
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        None


class BaseMeta(BaseAlgo, abc.ABC):

    def __init__(self,
                 name: str,
                 num_inner_algo: int,
                 computer_type: str = 'PC',
                 ):
        """
        Work file system for meta algorithm like:
            meta_name
                 - inner1
                   - cache
                   - log
                 - inner2
        """
        self.name = name
        self.num_inner_algo = num_inner_algo

        self._computer_type = computer_type

        prefix_name = 'inner'

        # create args pools and data pools
        self.args_pools = {}
        self.data_pools = {}

        # config path
        self.meta_json_path = 'meta'

        for i in range(num_inner_algo):
            inner_name = prefix_name + str(i)
            args_pool = ArgsPool(
                name=inner_name,
                level=1,
                computer_type=self._computer_type,
            )

            data_pool = DataPool(
                name=inner_name,
                level=1,
                computer_type=self._computer_type,
            )

            self.args_pools[inner_name] = args_pool
            self.data_pools[inner_name] = data_pool

    def init_pool(self):
        for inner_name, data_pool in self.data_pools.items():
            file_path = self.name + '/' + inner_name + '/' + self.meta_json_path
            data_pool_file = file_path + '/data_pool_config.json'
            args_pool_file = file_path + '/args_pool_config.json'

            # load json and create data pool unit
            data_pool_info_dict = json.load(open(data_pool_file))

            data_pool.create_buffer_from_dict_direct(data_pool_info_dict, inner_name)

            data_pool.set_pool_file(data_pool_file)

            # load json and create args pool unit
            args_pool_info_dict = json.load(open(args_pool_file))
            args_pool = self.args_pools[inner_name]
            args_pool.set_pool_file(args_pool_file)
            args_pool.create_buffer_from_dict_direct(args_pool_info_dict, inner_name)

            self.args_pools[inner_name] = args_pool
            self.data_pools[inner_name] = data_pool

    def read_data_pool(self):

        # the data pool will read directly from the file
        for inner_name, data_pool in self.data_pools.items():
            data_pool_info_dict = json.load(open(data_pool.pool_file))
            data_pool.read_shared_memory_from_dict_direct(data_pool_info_dict, inner_name)

    def close_data_pool(self):
        for inner_name, data_pool in self.data_pools.items():
            data_pool.close()

    def close_pool(self):
        for inner_name, data_pool in self.data_pools.items():
            data_pool.close()
            self.data_pools[inner_name].close()

    def optimize(self):
        pass

    @abc.abstractmethod
    def __optimize__(self):
        pass
