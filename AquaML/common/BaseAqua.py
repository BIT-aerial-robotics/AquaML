'''

2.1测试版，启动算法统一入口。
'''

from abc import ABC, abstractmethod
from AquaML.common.FileSystem import DefaultFileSystem, BaseFileSystem
from AquaML.common.Recoder import Recoder

class BaseAqua(ABC):
    """
    该类是算法启动器的基类，所有算法启动器都应该继承该类。

    在2.1版设计理念中， 多线程将由基类统一管理，算法启动由一个类管理。
    注意设计时候考虑tuner的使用，即在多线程中，该类能很好隔离一堆算法的干扰，各自运行各的。

    我们将一个完整运行的程序成为一个Aqua，一个Aqua可以包含多个Agent，也可以包含Aqua。
    多个learning agent通常用于多智能体算法中，多个Aqua通常用于调参。

    我们将从如下角度来设计该框架：

    1. Aqua与Agent的关系：
        每个Aqua至少有一个Agent
    2. Aqua与Aqua的关系：
        一个Aqua可以包含多个Aqua， 多线程时候，一个MPI Group对应一个Aqua， Aqua内部会自动分配这些通信组
    3. 文件系统：
        Aqua只有一套文件系统，即使Auqa包含多个Aqua，也只有一套文件系统，Auqa直接帮助agent管理文件系统
        Aqua能够自己创建文件夹，agent需要靠Aqua创建文件夹
    4. 数据传输系统：
        Auq会维护一个数据池子供自己的子Aqua和Agent使用，多线程时会自动共享数据池子


    相关接口：
    1. self._sub_aqua_dict: 存储sub Aqua agent 的dict, 作为Aqua去了解算法，分配检查一个接口， 
       每有一个新的Aqua和agent请添加到该dict中。
    
    
    """

    def __init__(self,
                 name,
                 level=0,
                 file_system:str or DefaultFileSystem='Default',
                 ):
        """

        Args:
            project_name (_type_): _description_
            level (int, optional): _description_. Defaults to 0.
            file_system (str, optional): _description_. Defaults to 'Default'.
        """

        self.name = name
        self.level = level

        # 创建文件系统
        if isinstance(file_system, str):
            if file_system == 'Default':
                self.file_system = DefaultFileSystem(name,
                                                     thr_level=level
                                                     )
            else:
                raise ValueError('file_system must be a str or DefaultFileSystem')
        
        elif issubclass(file_system, BaseFileSystem):
            self.file_system = file_system # 直接使用主Aqua的文件系统

        # 注意:所有的文件夹创建都由level 0的对象来创建

        # 指定接口

        # 存储sub Aqua agent 的dict
        self._sub_aqua_dict = {}

    def check(self):
        """
        检查Aqua的状态，是否可以启动
        """
        
        # 检查_sub_aqua_dict
        if len(self._sub_aqua_dict) == 0:
            raise ValueError('Aqua must have at least one sub Aqua or agent')
        
        # 检查名称和sub是否合法
        for _, value in self._sub_aqua_dict.items():
            if self.name + '_' not in value.name:
                raise ValueError('sub Aqua or agent name must contain Aqua name')

            value.check()
    
    def inita_folder(self):
        """
        初始化文件夹
        """
        
        # 添加_sub_aqua_dict中的文件夹
        for _, value in self._sub_aqua_dict.items():
            self.file_system.add_new(value.name)
            if issubclass(value, BaseAqua):
                value.inita_folder()
    
    def create_recoder(self):
        """
        创建记录器
        """

        # 获取记录器名称
        recoder_name = self.file_system.get_log_file
        log_file = self.file_system.log_root_path

        if self.level == 0:
            self.recoder = Recoder(
                log_folder=log_file,
                recoder_name=recoder_name
            )
        else:
            self.recoder = None
    
