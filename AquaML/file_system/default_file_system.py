from AquaML import coordinator
from AquaML.file_system.base_file_system import BaseFileSystem


@coordinator.registerFileSystem
class DefaultFileSystem(BaseFileSystem):
    """
    默认文件系统
    """

    def __init__(self, workspace_dir: str ):
        """
        初始化文件系统,
        Args:
            workspace_dir (str): 工作空间目录,绝对路径，一般建议以环境命名。
            create_first (bool, optional): 是否创建文件夹,在多进程系统中，只有主进程创建文件夹。默认为True. 
        """
        super().__init__(workspace_dir)
