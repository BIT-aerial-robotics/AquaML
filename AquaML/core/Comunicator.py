from mpi4py import MPI
from abc import ABC, abstractmethod
from AquaML.data.DataPool import DataPool
from AquaML.core.DataParser import DataInfo


class DataCollection:

    def __init__(self, name: str,
                 data_pool_info=None,
                 param_pool_info=None,
                 indicate_pool_info=None,
                 level: int = 0
                 ):
        """

        Communicator直接创建的数据集合，用于存储数据。

        param_pool和data_pool可以是一个空值

        Args:
            name (str): 数据集合的名称。
            data_pool_info (DataInfoorNone, optional): _description_. Defaults to None.
            param_pool_info (DataInfoorNone, optional): _description_. Defaults to None.
        """
        ###############################################
        # 创建数据池
        # TODO: 监督学习接口拓展
        ###############################################

        if data_pool_info is None:
            self.data_pool = None
        else:
            self.data_pool = DataPool(name=name, level=level)
            self.data_pool.multi_init(data_pool_info, type='buffer')

        if param_pool_info is None:
            self.param_pool = None
        else:
            self.param_pool = DataPool(name=name, level=level)
            self.param_pool.multi_init(param_pool_info, type='buffer')

        if indicate_pool_info is None:
            self.indicate_pool = None
        else:
            self.indicate_pool = DataPool(name=name, level=level)
            self.indicate_pool.multi_init(indicate_pool_info, type='buffer')

        ###############################################
        # data_clollection信息
        ###############################################
        self.name = name
        self.level = level

        if data_pool_info is None:
            self._data_pool_total_size = 0
        else:
            self._data_pool_total_size = data_pool_info.get_total_size

        if indicate_pool_info is None:
            self._indicate_pool_total_size = 0
        else:
            self._indicate_pool_total_size = indicate_pool_info.get_total_size

    @property
    def get_data_pool_size(self):
        return self._data_pool_total_size

    @property
    def get_indicate_pool_size(self):
        return self._indicate_pool_total_size

    @property
    def get_data_pool_dict(self):

        return_dict = {}
        if self.data_pool is None:
            raise ValueError("data_pool is None")

        for data_name, unit in self.data_pool.data_pool.items():
            return_dict[data_name] = unit.buffer

        return return_dict


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

    @abstractmethod
    def thread_level(self):
        """
        获取当前线程的层级
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

    @property
    def thread_level(self):
        if self.thread_id == 0:
            return 0
        else:
            return 1


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

    @property
    def thread_level(self):
        return 0


class ThreadManagerScheduler:
    def __init__(self):
        self.MPI = MPIThreadManager
        self.Single = SingleThreadManager


class CommunicatorBase(ABC):
    def __init__(self, thread_manager_info: dict):
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

        self.level = self.thread_manager.thread_level

        ###############################################
        # 数据块分配
        ###############################################

        self.data_pool_start_index = 0
        self.data_pool_end_index = 0

        self.indicate_pool_start_index = 0
        self.indicate_pool_end_index = 0

    def compute_start_end_index(self, data_pool_size: int, indicate_pool_size: int, worker_threads: int,
                                worker_id: int):
        """
        计算数据块的起始和结束索引。

        Args:
            data_pool_size (int): 数据池的大小。
            indicate_pool_size (int): 指示池的大小。
            worker_threads (int): worker的线程数。
            worker_id (int): worker的id。
        """

        data_pool_segment_length = data_pool_size // worker_threads
        indicate_pool_segment_length = indicate_pool_size // worker_threads

        self.data_pool_start_index = (worker_id - 1) * data_pool_segment_length
        self.data_pool_end_index = worker_id * data_pool_segment_length

        self.indicate_pool_start_index = (worker_id - 1) * indicate_pool_segment_length
        self.indicate_pool_end_index = worker_id * indicate_pool_segment_length

    def create_data_collection(self, name: str,
                               data_pool_info=None,
                               param_pool_info=None,
                               indicate_pool_info=None,
                               ):
        """
        创建数据集合。

        Args:
            name (str): 数据集合的名称。
            data_pool_info (DataInfoorNone, optional): 数据池的信息。 Defaults to None.
            param_pool_info (DataInfoorNone, optional): 参数池的信息。 Defaults to None.
            indicate_pool_info (DataInfoorNone, optional)：指示池的信息。 Defaults to None.
        """

        self._collection_data_fict[name] = DataCollection(
            name=name,
            data_pool_info=data_pool_info,
            param_pool_info=param_pool_info,
            indicate_pool_info=indicate_pool_info,
            level=self.level
        )

    def get_data_pool_dict(self, agent_name: str):
        """
        获取数据集合中的数据池。

        Args:
            agent_name (str): agent的名称。
        """
        return self._collection_data_fict[agent_name].data_pool.get_numpy_dict()

    def get_data(self, agent_name: str):
        """
        获取数据集合中的数据。

        Args:
            agent_name (str): agent的名称。
            data_name (str): 数据集合的名称。
        """
        return self._collection_data_fict[agent_name].get_data_pool_dict

    def get_param(self, agent_name: str, param_name: str):
        """
        获取数据集合中的参数。

        Args:
            agent_name (str): agent的名称。
            param_name (str): 参数的名称。
        """
        return self._collection_data_fict[agent_name].param_pool.get_data(param_name)

    def store_data(self, agent_name: str, data_name: str, data, start_index, end_index):
        """
        存储数据集合中的数据。

        Args:
            agent_name (str): agent的名称。
            data_name (str): 数据集合的名称。
            data ([type]): 数据。
        """

        self._collection_data_fict[agent_name].data_pool.store_sequence(data_name, data, start_index, end_index)

    def store_data_dict(self, agent_name: str, data_dict: dict, start_index, end_index):

        for data_name, data in data_dict.items():
            self.store_data(agent_name, data_name, data, start_index, end_index)

    def store_indicate_dict(self, agent_name: str, indicate_dict: dict, index, pre_fix=None):

        if pre_fix is not None:
            indicate_dict = {pre_fix + k: v for k, v in indicate_dict.items()}

        self._collection_data_fict[agent_name].indicate_pool.store(indicate_dict, index-1)

    def store_param(self, agent_name: str, param_name: str, param):
        """
        存储数据集合中的参数。

        Args:
            agent_name (str): agent的名称。
            param_name (str): 参数的名称。
            param ([type]): 参数。
        """
        self._collection_data_fict[agent_name].param_pool.store_all(param_name, param)

    def get_level(self):
        return self.level

    def get_data_pool_size(self, agent_name: str):
        """
        获取数据集合中的数据池的大小。

        Args:
            agent_name (str): agent的名称。
        """
        return self._collection_data_fict[agent_name].get_data_pool_size


class Communicator(CommunicatorBase):
    def __init__(self, thread_manager_info: dict, level: int = 0):
        super().__init__(
            thread_manager_info=thread_manager_info,
        )