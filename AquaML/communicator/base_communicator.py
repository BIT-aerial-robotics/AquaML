from loguru import logger
from abc import ABC, abstractmethod
import sys


class BaseCommunicator(ABC):
    """
    所有通信类的基类。
    """

    def __init__(self):
        """
        初始化通信类。
        """

        self.rank_ = None  # 进程ID
        self.size_ = None  # 进程数量

    def configLogger(self):
        """
        配置日志。日志输出时按照进程ID进行区分。
        """

        if self.rank_ is None:
            logger.error("rank is not set!")
            raise ValueError("rank is not set!")

        # colorize = {
        #     "INFO": "green",
        #     "WARNING": "yellow",
        #     "ERROR": "red",
        #     "CRITICAL": "bold red on white",
        #     "SUCCESS": "green",
        #     "DEBUG": "blue",
        # }
        logger_format = "<level> Rank" + \
            str(self.rank_) + \
            " {time:YYYY-MM-DD HH:mm:ss:ms} | {level} | {name}:{function}:{line} - {message} </level>"
        logger.add(sys.stderr, format=logger_format)

    @abstractmethod
    def initDataUnitPolicy(self):
        """
        初始化数据单元策略。
        """

        raise NotImplementedError