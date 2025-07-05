import numpy as np
from AquaML import aqua_tool

class LossTracker:
    def __init__(self):
        self.loss_dict = {}

    def add_data(self, loss_dict: dict, prefix: str = ''):

        if len(prefix) > 0:
            prefix = prefix + '/'

        for key, value in loss_dict.items():

            all_name = prefix + key

            if all_name not in self.loss_dict:
                self.loss_dict[all_name] = []
                
            self.loss_dict[all_name].append(aqua_tool.convert_numpy_fn(value))


    def reset(self):
        self.loss_dict = {}

    def get_data(self):
        loss_dict = {}
        for key, value in self.loss_dict.items():
            # array = deepcopy(value)
            loss_dict[key + '_max'] = np.max(value)
            loss_dict[key + '_min'] = np.min(value)
            loss_dict[key] = np.mean(value)

            if 'reward' in key:
                loss_dict[key + '_std'] = np.std(value)

            if 'indicate' in key:
                loss_dict[key + '_std'] = np.std(value)

        # 获取data以后自动释放内存
        self.reset()
        return loss_dict


class DataSetTracker:
    """
    用于追踪多线程或者需要分段处理数据，最后将数据进行汇总为标准训练集
    """

    def __init__(self):
        self.data_dict = {}

    def add_data(self, data_dict: dict, prefix: str = ''):

        if len(prefix) > 0:
            prefix = prefix + '/'

        for key, value in data_dict.items():

            all_name = prefix + key

            if all_name not in self.data_dict:
                self.data_dict[all_name] = []
            self.data_dict[all_name].append(value)

    def gett_data(self):
        data_dict = {}
        for key, value in self.data_dict.items():
            data_dict[key] = np.vstack(value)

        return data_dict


