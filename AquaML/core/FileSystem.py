'''
文件系统，用于管理文件、log存储的地方

在多个Aqua实列中，可以有效的分配文件存储的位置
'''

from abc import ABC, abstractmethod
import os
import datetime


class BaseFileSystem(ABC):
    """

    所有的算法(Aqua)中基础文件系统如下所示：
    -project name
        -cache
            -agent_1
                cache_file
            -agent_2
                cache_file
            ...
        -history_model
            -agent_1
                -history_folder
                    history_file
            -agent_2
                -history_folder
                    history_file
            ...
        -log
            log_file

    用户可以自定义文件系统，但是必须继承该类, 需要遵循以下规则：

    1. 添加新的文件夹时，按照：
        self.name_root_path = os.path.join(project_name, 'name')
        self.mkdir(self.name_root_path)

        # 添加相应的dict
        self.name_path_dict = {
            'root_path': self.name_root_path,
        }

        # 在_add_function_dict中添加相应的函数
        self._add_function_dict['name'] = self.add_name_path

    2. 添加新的文件夹时，需要添加相应的函数，如下所示：
        def add_name_path(self, name:str):
            self.name_path_dict[name] = os.path.join(self.name_root_path, name)
            self.mkdir(self.name_path_dict[name])
        

    """

    def __init__(self, project_name,
                 aqua_level=1,
                 thr_level=0,
                 log_name=None,
                 ):
        """
        初始化文件系统

        :param project_name: 项目名称
        :param aqua_level: Aqua的级数
        :param thr_level: 线程级数
        """
        self.project_name = project_name
        self.aua_level = aqua_level
        self.thr_level = thr_level

        # 分别记录cache、history_model、log的路径
        self.cache_root_path = os.path.join(project_name, 'cache')
        self.history_model_root_path = os.path.join(project_name, 'history_model')
        self.log_root_path = os.path.join(project_name, 'log')

        self.mkdir(self.cache_root_path)
        self.mkdir(self.history_model_root_path)
        self.mkdir(self.log_root_path)

        self.cahce_path_dict = {
            'root_path': self.cache_root_path,
        }

        self.history_model_path_dict = {
            'root_path': self.history_model_root_path,
        }

        self.log_path_dict = {
            'root_path': self.log_root_path,
        }

        self._add_function_dict = {
            'cache': self.add_cache_path,
            'history_model': self.add_history_model_path,
            'log': self.add_log_path,
        }

        # 确定log的名称
        if log_name is None:
            log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            log_name = log_name

        self.log_path = os.path.join(self.log_root_path, log_name)

    def add_cache_path(self, name: str):

        """
        添加cache的路径

        :param name: 加入的名词
        :return:
        """

        self.cahce_path_dict[name] = os.path.join(self.cache_root_path, name)
        self.mkdir(self.cahce_path_dict[name])

    def add_history_model_path(self, name: str):

        """
            添加history_model的路径
    
            :param name: 加入的名词
            :return:
            """

        self.history_model_path_dict[name] = os.path.join(self.history_model_root_path, name)
        self.mkdir(self.history_model_path_dict[name])

    def add_log_path(self, name: str):

        """
            添加log的路径
    
            :param name: 加入的名词
            :return:
            """

        self.log_path_dict[name] = os.path.join(self.log_root_path, name)
        self.mkdir(self.log_path_dict[name])

    def add_new(self, name: str, filter: tuple = ('log',)):

        """
        添加新的路径

        :param name: 加入的名词
        :param filter: 过滤器，用于过滤不需要的文件夹
        :return:
        """
        for key in self._add_function_dict.keys():
            if key not in filter:
                self._add_function_dict[key](name)

    def mkdir(self, path: str):
        """
        创建文件夹

        :param path: 文件夹路径
        :return:
        """

        current_path = os.getcwd()
        path = os.path.join(current_path, path)
        if not os.path.exists(path):
            if self.thr_level == 0:
                os.makedirs(path)

    def get_cache_path(self, name: str):
        """
        获取cache的路径

        :param name: 名称
        :return: 路径
        """
        return self.cahce_path_dict[name]

    def get_history_model_path(self, name: str):
        """
        获取history_model的路径

        :param name: 名称
        :return: 路径
        """
        return self.history_model_path_dict[name]

    def get_log_path(self, name: str):
        """
        获取log的路径

        :param name: 名称
        :return: 路径
        """
        return self.log_path_dict[name]

    @property
    def get_log_file(self):
        """
        获取log的文件

        :param name: 名称
        :return: 文件
        """
        return self.log_path


class DefaultFileSystem(BaseFileSystem):

    def __init__(self, project_name, thread_level, aqua_level=1):
        super().__init__(project_name, aqua_level, thread_level)
