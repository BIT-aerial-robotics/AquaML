import abc


class DataInfoBase(abc.ABC):
    """
    args base class.
    """

    def __init__(self):
        self._total_length = None
        self._data_dic = None

    @property
    def data_dic(self):
        return self._data_dic
    
    @property
    def total_length(self):
        return self._total_length
