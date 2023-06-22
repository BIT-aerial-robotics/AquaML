from abc import ABC, abstractmethod

class BaseBuffer(ABC):
    """
    Abstract base class for buffer implementations.

    This class defines the interface for buffer implementations, which are used to store and retrieve data in a
    sequential manner. Subclasses must implement the `add` and `get` methods to add data to the buffer and retrieve
    data from the buffer, respectively.

    Attributes:
        capacity (int): The maximum number of items that the buffer can hold.
    """

    def __init__(self, capacity: int, data_names: list):
        """
        Initializes a new instance of the `BaseBuffer` class.

        Args:
            capacity (int): The maximum number of items that the buffer can hold.
        """
        self.capacity = capacity

        # 创建dict，用于存储数据
        self.data = {}

        # 创建list，用于存储数据名称
        for data_name in data_names:
            self.data[data_name] = []
        
        self.data_names = data_names

        # 参数
        self.capacity_count = 0
    
    def append(self, data: dict):
        """
        Appends data to the buffer.

        所有的数据长度必须保持一致，并且一batch size的形式存储进去

        Args:
            data (dict): The data to be appended to the buffer.
        """

        # 数据处理
        data_dict = self._process_data(data)

        # 数据存储
        if self.capacity_count < self.capacity:
            for data_name in self.data_names:
                self.data[data_name].append(data_dict[data_name])
            self.capacity_count += 1
        else:
            Index = self.capacity_count % self.capacity
            for data_name in self.data_names:
                self.data[data_name][Index] = data_dict[data_name]



    def _process_data(self, data: dict):
        """
        Processes the data in the buffer.

        This method is called by the `get` method to process the data in the buffer before it is returned. This method
        can be overridden in subclasses to perform additional processing on the data.

        Return like this:
        {'grad_mask':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],}
        """
        raise NotImplementedError


    