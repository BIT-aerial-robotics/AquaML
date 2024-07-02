"""
下一代版本的DataModule，下一代该模块会默认启动运行。
"""

from AquaML.core.DataUnit import DataUnit
from AquaML import logger,settings
from AquaML.core.DataInfo import DataInfo
from AquaML.core.DataList import DataList

class DataModule:
    """
    DataModule是一个数据物流中心，在框架里面将生产者产生的数据，有条理的归类。
    所有的消费者都可以通过数据名称查询到对应的数据。
    
    DataModule除了管理基于numpy共享内存的数据外，还可以管理list。
    
    TODO:当前list只能在单进程中进行数据管理，不能跨进程访问。
    
    DataModule设计理念：
    
    ############################
    
    生产者
    
    —————————
    
    ｜DataModule｜
    
    —————————
    
    消费者
    
    ############################
    
    在AquaML中，为了简化数据流的设计，使用者只需要暴力的将生产的数据放入DataModule中，
    在任何地方的消费者都可以通过数据名称查询到对应的数据。
    
    我们也会逐渐优化这部分的使用体验。
    """
    def __init__(self):
        self._memory_path = None
        
        
        # AquaML框架必须程序运行状态
        # 用于存储系统空间，name+units命名
        self._system_units = {}
        
        
        # 存储memory管理空间
        self._units_names = {
            'system':self._system_units
        } 
        
        self._list_names = {}
        
        ############################
        # RL部分特殊接口
        # 部分数据不方便使用DataUnit和DataList进行管理，直接在DataModule中进行注册。
        ############################
        self.rl_dict = {} # 用于存储RL部分的数据,如reward。
        
    def init(self):
        """
        初始化DataModule。
        """
        self._memory_path = settings.memory_path
    
    def config_algo(self, algo_name:str):
        """
        配置算法的二级数据模块
        
        Args:
            algo_name (str): 算法名称。
        """
        
        self._units_names[algo_name] = {}
        self._list_names[algo_name] = {}
        
      
    def create_data_unit(self, 
                         unit_name:str,
                         units_name:str,
                         unit_info:dict,
                         exist: bool = False,
                         org_shape=None
                         ):
        """
        创建共享内存数据单元。
        
        Args:
            unit_name (str): 数据单元的名称。
        """
        
        # 检测unit_names是否存在
        if units_name not in self._units_names:
            logger.error(f'{units_name} not exists')
            
        if unit_name in self._units_names[units_name]:
            logger.warning(f'{unit_name} already exists in {units_name}')
        else:
            name = self._memory_path + '_' + units_name + '_' + unit_name
            self._units_names[units_name][unit_name] = DataUnit(
                name=name,
                unit_info=unit_info,
                exist=exist,
                org_shape=org_shape
            )
            setattr(self, name, self._units_names[units_name][unit_name])
            
    def create_data_list(self,
                         lists_name:str,
                         list_name:str,
                         list_info:dict,
    ):
        """
        创建数据列表。
        
        Args:
            lists_name (str): 数据单元名称。
            list_name (str): 数据列表名称。
            list_info (dict): 数据列表信息。
        """
        
        if lists_name not in self._list_names:
            logger.error(f'{lists_name} not exists')
            
        if list_name in self._list_names[lists_name]:
            logger.warning(f'{list_name} already exists in {lists_name}')
        else:
            self._list_names[lists_name][list_name] = DataList(
                name=list_name,
                list_info=list_info
            )
            setattr(self, list_name, self._list_names[lists_name][list_name])
            
    def create_data_unit_from_data_info(self, 
                                       data_info:DataInfo,
                                       units_name:str,
                                       size:int,
                                       exist: bool = False,
                                       using_org_shape:bool = False
                                       ):
        """
        从data_info创建数据单元。
        
        Args:
            data_info (DataInfo): 数据信息。
            units_name (str): 数据单元名称。
            size (int): 数据单元大小。
        """
        
        unit_infos = data_info.generate_unit_infos(size=size)
        
        for unit_name, unit_info in unit_infos.items():
            org_shape = data_info.last_shape_dict[unit_name] if using_org_shape else None
            self.create_data_unit(
                unit_name=unit_name,
                units_name=units_name,
                unit_info=unit_info,
                exist=exist,
                org_shape=org_shape
            )
    
    def create_data_list_from_data_info(self,
                                       data_info:DataInfo,
                                       lists_name:str,
                                       size:int,
                                       ):
        """
        从data_info创建数据列表。
        
        Args:
            data_info (DataInfo): 数据信息。
            lists_name (str): 数据列表名称。
            size (int): 数据列表大小。
        """
        list_infos = data_info.generate_unit_infos(size=size)
        
        for list_name, list_info in list_infos.items():
            self.create_data_list(
                lists_name=lists_name,
                list_name=list_name,
                list_info=list_info
            )
            
    ############################
    # 数据查询接口
    ############################
    def query_data_unit(self, 
                        unit_name:str,
                        units_name:str
                        )->DataUnit:
        """
        查询数据单元。
        
        Args:
            unit_name (str): 数据单元名称。
            units_name (str): 数据单元集合名称
        """
        
        if units_name not in self._units_names:
            logger.warning(f'{units_name} not exists')
            
            
        return self._units_names[units_name][unit_name]
    
    def query_data_list(self,
                        list_name:str,
                        lists_name:str
                        )->DataList:
        """
        查询数据列表。
        
        Args:
            list_name (str): 数据列表名称。
            lists_name (str): 数据列表集合名称。
        """
        
        if lists_name not in self._list_names:
            logger.warning(f'{lists_name} not exists')
            
            
        # if list_name not in self._list_names[lists_name]:
        #     logger.error(f'{list_name} not exists in {lists_name}')
            
        return self._list_names[lists_name][list_name]
    
    def query_data(self,
                   name:str,
                   set_name:str
                   )->list:
        """
        在unit和list中查询数据。

        Args:
            name (str): 数据名称。
            set_name (str): 数据集合名称。
        Return:
            list: 返回查询到的数据。
        """
        
        return_list = []
        
        # 查询unit
        if set_name in self._units_names:
            try:
                return_list.append(self._units_names[set_name][name])
            except:
                pass
        
        # 查询list
        if set_name in self._list_names:
            try:
                return_list.append(self._list_names[set_name][name])
            except:
                pass
        
        return return_list
        
        
        
    @property
    def system_units(self):
        """
        返回系统数据单元。
        """
        return self._system_units
    
    @property
    def units(self)->dict:
        """
        返回所有数据单元。
        """
        return self._units_names
    
    @property
    def lists(self)->dict:
        """
        返回所有数据列表。
        """
        return self._list_names
        
        
        