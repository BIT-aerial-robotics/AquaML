import copy
# from typing import overload

class DataInfo:
    def __init__(self, names: tuple = None, shapes: tuple = None, dtypes = None):
        """
        初始化DataInfo类的实例。

        :param names: 一个包含名称的元组，每个名称对应一个数据集。默认为None。
        :param shapes: 一个包含形状的元组，每个形状对应一个数据集。默认为None。
        :param dtypes: 一个包含数据类型的元组或单个数据类型，每个数据类型对应一个数据集。默认为None。

        如果names、shapes和dtypes都为None，将创建一个空的DataInfo实例。
        如果提供了参数，将根据参数创建一个DataInfo实例。
        """

        # 如果names为None，设置为一个空元组
        if names is None:
            names = tuple()

        # 如果shapes为None，设置为一个空字典
        if shapes is None:
            shapes = {}

        # 如果dtypes为None，设置为一个空字典
        if dtypes is None:
            dtypes = {}

        # 如果提供了names，根据names和shapes创建一个字典，否则创建一个空字典
        self.shape_dict = dict(zip(names, shapes)) if names else {}
        self.names = names

        # 如果dtypes是一个元组，根据names和dtypes创建一个字典，否则为每个name分配dtypes
        if isinstance(dtypes, tuple):
            self.type_dict = dict(zip(names, dtypes)) if names else {}
        else:
            self.type_dict = dict()
            for key in names:
                self.type_dict[key] = dtypes

        # 如果提供了names，复制shape_dict，否则创建一个空字典
        self.last_shape_dict = copy.deepcopy(self.shape_dict) if names else {}
                
        

    def add_info(self, name: str, shape, dtype):
        """add info.

        Args:
            name (str): name of the data.
            shape (tuple): shape of the data.
            dtype (type): dtype of the data.
        """
        self.shape_dict[name] = shape
        self.type_dict[name] = dtype

        # add element to names
        names = list(self.names)
        names.append(name)
        self.names = tuple(names)
        

    
    def add_axis0_shape(self, value: int):
        """
        为数据的维度0添加指定数值。
        如已有shape为(1,2,3),运行add_axis0_shape(2)后，shape变为(2,1,2,3)。
        

        Args:
            value (int): 指定数值。
        """
        
        self.last_shape_dict = copy.deepcopy(self.shape_dict)
        
        for key, shape in self.shape_dict.items():
            self.shape_dict[key] = (value,*shape)
            
    
    def generate_unit_infos(self, size: int=1):
        """
        生成数据单元或者list的信息。
        
        Returns:
            dict: 数据单元的信息。
            size (int, optional): 数据单元的大小。 Defaults to 1.
        """
        
        unit_infos = {}
        
        for name in self.names:
            unit_infos[name] = {
                'shape': self.shape_dict[name],
                'dtype': self.type_dict[name],
                'size': size
            }
            
        return unit_infos
    
    # 插入前缀并生成新的DataInfo
    def insert_prefix(self, prefix: str):
        """
        为names添加前缀并生成新的DataInfo。
        
        Args:
            prefix (str): 前缀。
        
        Returns:
            DataInfo: 新的DataInfo。
        """
        
        names = tuple([prefix+name for name in self.names])
        
        return DataInfo(names, self.shape_dict.values(), self.type_dict.values())
        
        
    def __add__(self, other):
        """Add two DataInfo.

        Args:
            other (DataInfo): another DataInfo.

        Returns:
            DataInfo: new DataInfo.
        """
        names = (*self.names, *other.names)
        shapes = (*self.shape_dict.values(), *other.shape_dict.values())
        dtypes = (*self.type_dict.values(), *other.type_dict.values())

        return DataInfo(names, shapes, dtypes)
    
    # def __call__(self, ):
    #     # 查询信息的时候返回unit_infos
    #     return self.generate_unit_infos()
if __name__ == "__main__":
    
    data_info = DataInfo(('a', 'b'), ((1,2,3), (4,5,6)), (int, float))
    
    data_info.add_axis0_shape(2)
    
    next_data_info = data_info.insert_prefix('next_')
    
    next_data_info.add_info('mask', shape=(1,), dtype=bool)
    
    print(next_data_info.generate_unit_infos(1))
    
    void_data_info = DataInfo()
    
    print(void_data_info.shape_dict)