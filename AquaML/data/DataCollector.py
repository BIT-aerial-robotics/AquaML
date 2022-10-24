from AquaML.data.DataUnit import DataUnit
import numpy as np


class DataCollector:
    def __init__(self, data_dic: dict, total_length: int, name_prefix: str, share_memory=False):
        """
        create data pool according to your setting.
        :param data_dic:(dict) The data you will use in the later training. The dict store the single data shape of every data.
        :param total_length:(int) The total number of your samples.
        :param name_prefix:(str) When you create your training task, you name your training task. We also call it as work space
        :param share_memory:(str)
        """
        self.data_pool = dict()  # store every DataPool
        self.name_prefix = name_prefix

        for key, value in data_dic.items():
            self.data_pool[key] = DataUnit(name=name_prefix + '_' + key, shape=value, total_length=total_length,
                                           share_memory=share_memory)

    def store(self, data: dict, index):
        """
        store data in data pool.

        :param data:
        :param index:
        :return:
        """

        for key, value in data.items():
            self.data_pool[key].store(value, index)

    def close(self):
        for key, value in self.data_pool.items():
            value.close()

    def slice_data(self, name: list, start: int, end: int):

        ret_dic = dict()
        for key in name:
            ret_dic[key] = self.data_pool[key].data_block(start, end)

        return ret_dic

    def save_data(self):
        for key, value in self.data_pool:
            value.save(self.name_prefix)

    def load_data(self):
        for key, value in self.data_pool:
            value.load_date(self.name_prefix)
