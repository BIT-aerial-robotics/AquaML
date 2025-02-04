import numpy as np
from typing import Any


class DictConcat:
    def __init__(self,):
        '''
        Initialize the DictConcat class.

        This class is used to concatenate the data in the dictionary.
        '''

        self.dict_ = {}
        self.init_flag_ = False

    def reset(self,):
        if not self.init_flag_:
            return

        # Reset every element to void list.
        for key in self.dict_:
            self.dict_[key] = []

    def append(self, data: dict):
        if not self.init_flag_:
            self.init_flag_ = True
            for key in data:
                self.dict_[key] = []
        for key in data:
            self.dict_[key].append(data[key])

    def getConcatData(self, concat_axis: int = 0):
        '''
        concatenate the data in the dictionary.

        :param concat_axis: The axis to concatenate the data.
        :type concat_axis: int
        :return: The concatenated data.
        :rtype: dict
        '''

        concat_data = {}
        for key in self.dict_:
            concat_data[key] = np.concatenate(
                self.dict_[key], axis=concat_axis)
        return concat_data


class ScalarConcat:
    def __init__(self,):
        '''
        Initialize the ScalarConcat class.

        This class is used to concatenate the scalar data.
        '''

        self.data_ = []
        self.init_flag_ = False

    def reset(self,):
        if not self.init_flag_:
            return

        # Reset every element to void list.
        self.data_ = []

    def append(self, data: Any):
        if not self.init_flag_:
            self.init_flag_ = True
        self.data_.append(data)

    def getConcatData(self, concat_axis: int = 0):
        '''
        concatenate the data in the list.

        :param concat_axis: The axis to concatenate the data.
        :type concat_axis: int
        :return: The concatenated data.
        :rtype: np.ndarray
        '''

        return np.concatenate(self.data_, axis=concat_axis)
