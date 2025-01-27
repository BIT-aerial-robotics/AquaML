import os
from AquaML import logger


def mkdir(path: str):
    """
    创建文件夹

    :param path: 文件夹路径
    :return:
    """

    # current_path = os.getcwd()
    # path = os.path.join(current_path, path)
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Create folder {path}")
        return True
    else:
        logger.info(f"Folder {path} already exists")
        return False
