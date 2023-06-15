from mpi4py import MPI
from abc import ABC, abstractmethod
from AquaML.data.DataPool import DataPool
from AquaML.data.DataParser import DataInfo

class DataCollection:

    def __init__(self, name:str, 
                 data_pool_info: DataInfo or None = None,
                 param_pool_info: DataInfo or None = None,
                 level:int = 0
                 ):
        """

        Communicator直接创建的数据集合，用于存储数据。

        param_pool和data_pool可以是一个空值

        Args:
            name (str): 数据集合的名称。
            data_pool_info (DataInfoorNone, optional): _description_. Defaults to None.
            param_pool_info (DataInfoorNone, optional): _description_. Defaults to None.
        """

        # TODO: 监督学习接口拓展

        if data_pool_info is None:
            self.data_pool = None
        else:
            self.data_pool = DataPool(name=name, level=level)
            self.data_pool.multi_init(data_pool_info,type='buffer')

        if param_pool_info is None:
            self.param_pool = None
        else:
            self.param_pool = DataPool(name=name, level=level)
            self.param_pool.multi_init(param_pool_info,type='buffer')

class MultiThreadManagerBase(ABC):
    
    @abstractmethod
    def get_total_threads(self):
        """
        获取总的可用线程数
        """
    
    @abstractmethod
    def get_thread_id(self):
        """
        获取当前线程的id
        """

class MPIThreadManager(MultiThreadManagerBase):
    """

    MPI多线程管理器,支持Group通信。
    """
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.thread_id = comm.Get_rank()
        self.total_threads = comm.Get_size()

    @property
    def get_thread_id(self):
        return self.thread_id

    @property
    def get_total_threads(self):
        return self.total_threads
    
    def Barrier(self):
        self.comm.Barrier()

class SingleThreadManager(MultiThreadManagerBase):
    """
    单线程管理器，用于单线程运行。
    """
    def __init__(self):
        self.thread_id = 0
        self.total_threads = 1

    @property
    def get_thread_id(self):
        return self.thread_id

    @property
    def get_total_threads(self):
        return self.total_threads

    def Barrier(self):
        pass

class ThreadManagerScheduler:
    def __init__(self):
        self.MPI = MPIThreadManager
        self.Single = SingleThreadManager

class CommunicatorBase(ABC):
    def __init__(self, thread_manager_info:dict, level:int = 0):
        """
        通信器的基类，用于管理数据同步等操作。

        Args:
            thread_manager_info (dict): 线程管理器的信息，用于创建线程管理器。格式为：
            {
                "type": "MPI",
                "args": {
                    "comm": MPI.COMM_WORLD,
                }
            }
        """
        Scheduler = ThreadManagerScheduler()

        self._collection_data_fict = {}
        

        self.thread_manager = getattr(Scheduler, thread_manager_info["type"])(**thread_manager_info["args"])

        self.level = level

    def create_data_collection(self, name:str, 
                               data_pool_info: DataInfo or None = None,
                               param_pool_info: DataInfo or None = None):
        """
        创建数据集合。

        Args:
            name (str): 数据集合的名称。
            data_pool_info (DataInfoorNone, optional): _description_. Defaults to None.
            param_pool_info (DataInfoorNone, optional): _description_. Defaults to None.
        """
        
        self._collection_data_fict[name] = DataCollection(
            name=name,
            data_pool_info=data_pool_info,
            param_pool_info=param_pool_info,
            level=self.level
        )

    
    def get_data(self, agent_name:str, data_name:str):
        """
        获取数据集合中的数据。

        Args:
            agent_name (str): agent的名称。
            data_name (str): 数据集合的名称。
        """
        return self._collection_data_fict[agent_name].data_pool.get_data(data_name)
    
    def get_param(self, agent_name:str, param_name:str):
        """
        获取数据集合中的参数。

        Args:
            agent_name (str): agent的名称。
            param_name (str): 参数的名称。
        """
        return self._collection_data_fict[agent_name].param_pool.get_data(param_name)
    
    def store_data(self, agent_name:str, data_name:str, data, start_index, end_index):
        """
        存储数据集合中的数据。

        Args:
            agent_name (str): agent的名称。
            data_name (str): 数据集合的名称。
            data ([type]): 数据。
        """

        self._collection_data_fict[agent_name].data_pool.store_sequence(data_name, data, start_index, end_index)

    def store_param(self, agent_name:str, param_name:str, param):
        """
        存储数据集合中的参数。

        Args:
            agent_name (str): agent的名称。
            param_name (str): 参数的名称。
            param ([type]): 参数。
        """
        self._collection_data_fict[agent_name].param_pool.store_all(param_name, param)

    

class Communicator(CommunicatorBase):
    def __init__(self, thread_manager_info: dict, level: int = 0):
        super().__init__(
            thread_manager_info=thread_manager_info,
            level=level
        )