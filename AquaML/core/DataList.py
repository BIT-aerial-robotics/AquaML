# from AquaML import logger

class DataList(list):
    """
    用于存储list数据。
    
    当前不能够跨进程访问。
    
    当前按需根据进程需求来创建。
    """
    def __init__(self,
                 name: str,
                 list_info: dict,
    ):
        """
        创建DataList。

        Args:
            name (str): DataList的名称。
            list_info (dict): DataList的信息。
        """
        
        super(DataList, self).__init__()
        self._name = name
        self._max_size = list_info['size']
        self._dtype = list_info['dtype']
        self._shape = list_info['shape']
        self._org_shape = list_info['shape']
        
        self._current_size = 0
        
        # logger.info(f'Create DataList {name}')
        
    def append(self, data):
        """
        添加数据。
        
        Args:
            data: 数据。
        """
        
        if self._current_size >= self._max_size:
            raise ValueError('DataList is full')
        
        super(DataList, self).append(data)
        
        self._current_size += 1
    
    ########################################################
    # 通用接口
    ########################################################
    def reset(self):
        """
        重置DataList。
        """
        
        self.clear()
        self._current_size = 0
    
    def get_data(self):
        """
        获取DataList的数据。
        
        Returns:
            list: DataList的数据。
        """
        
        return self
    
    @property
    def name(self):
        return self._name
    
    @property
    def max_size(self):
        return self._max_size
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def shape(self):
        return self._shape
        
if __name__ == '__main__':
    test = DataList(name='test', max_size=2, dtype='float32', shape=(2,))
    
    test.append(1.0)
    test.append(2.0)
    # test.reset()
    # test.append(3.0)
    
    test.reset()
    print(test.get_data())
        
        