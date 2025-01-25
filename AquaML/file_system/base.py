'''
这里将简化FileSystem的设计，对文件系统作进一步优化，和明确的规定。
与之类似功能的有DataModule用于管理内存模块。

并且自动管理log文件。
'''

from AquaML import logger
from abc import ABC, abstractmethod
from AquaML.utils.tool import mkdir

class BaseFileSystem(ABC):
    """
    FileSystem用于管理文件系统。
    
    每个任务的文件系统将如下所示：
    - project_name
        - data_time/pointed_name/algo_name
            - cache
            - data_unit
            - history_model
            - log
        - logger
    """
    def __init__(self, project_name:str,create_first: bool = True, ):
        """
        初始化文件系统,

        Args:
            project_name (str): 项目名称.
            create_first (bool, optional): 是否创建文件夹,在多进程系统中，只有主进程创建文件夹。默认为True.
        """
        self.projecct_name_ = project_name # 项目名称，一般建议以环境命名
        
        self.create_first_ = create_first # 是否创建文件夹
        
        
    
    def initFolder(self):
        """
        初始化文件夹
        """
        
        if self.create_first_:
            logger.info(f"Init folder for {self.projecct_name_}")
        
            # 创建文件夹
            mkdir(self.projecct_name_)
            
    
    

