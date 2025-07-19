'''
这里将简化FileSystem的设计，对文件系统作进一步优化，和明确的规定。
与之类似功能的有DataModule用于管理内存模块。

并且自动管理log文件。
'''

from loguru import logger
from abc import ABC, abstractmethod
from AquaML.utils.tool import mkdir
import os
import time
import yaml
import torch
import pickle
import json
from typing import Any, Dict, Optional


class BaseFileSystem(ABC):
    """
    FileSystem用于管理文件系统。

    每个任务的文件系统将如下所示：
    - workspace_dir
        - runner_name
            - cache
            # - data_unit
            - history_model
            - log
        - logger
    """

    def __init__(self, workspace_dir: str):
        """
        初始化文件系统,

        Args:
            workspace_dir (str): 工作空间目录,绝对路径，一般建议以环境命名。
        """
        self.workspace_dir_ = workspace_dir  # 项目名称，一般建议以环境命名

        # 存储路径
        self.logger_path_ = os.path.join(
            self.workspace_dir_, "logger")  # logger文件夹路径

        self.runner_dir_dict_ = {}  # runner文件夹路径字典

        # 初始化文件夹
        # self.initFolder()

    def initFolder(self):
        """
        初始化文件夹
        """

        # if self.create_first_:
        logger.info(f"Init folder for {self.workspace_dir_}")

        # 创建文件夹workspace_dir
        mkdir(self.workspace_dir_)

        # 创建logger文件夹
        mkdir(self.logger_path_)

        # 初始化logger存储
        logger.info(f"Init logger for {self.workspace_dir_}")

        # 按照时间创建日志文件
        logger_file_name = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"

        log_level = os.environ.get("AQUAML_LOG_LEVEL", "INFO").upper()

        logger.add(os.path.join(self.logger_path_,
                   logger_file_name), rotation="100 MB", level=log_level)

    def configRunner(self, runner_name: str, create_first: bool = True):
        """
        注册runner文件夹

        Args:
            runner_name (str): runner名称
            create_first (bool, optional): 是否创建文件夹,默认为True.
        """

        runner_dir_dict = {
            # 缓存文件夹
            "cache": os.path.join(self.workspace_dir_, os.path.join(runner_name, "cache")),
            # 历史模型文件夹
            "history_model": os.path.join(self.workspace_dir_, os.path.join(runner_name, "history_model")),
            # 日志文件夹
            "log": os.path.join(self.workspace_dir_, os.path.join(runner_name, "log")),
            # 数据文件夹
            "data_unit": os.path.join(self.workspace_dir_, os.path.join(runner_name, "data_config")),
        }

        if create_first:
            # 创建文件夹
            for _, path in runner_dir_dict.items():
                mkdir(path)

        # 注册该runner的文件夹系统
        self.runner_dir_dict_[runner_name] = runner_dir_dict
        logger.info("successfully register runner {}".format(runner_name))

    def queryHistoryModelPath(self, runner_name: str) -> str:
        """
        查询历史模型文件夹路径
        Args:
            runner_name (str): runner名称
        Returns:
            str: 历史模型文件夹路径
        """

        try:
            return self.runner_dir_dict_[runner_name]["history_model"]
        except KeyError:
            logger.error("runner {} not registered".format(runner_name))
            raise KeyError("runner {} not registered".format(runner_name))

    def queryCachePath(self, runner_name: str) -> str:
        """
        查询缓存文件夹路径
        Args:
            runner_name (str): runner名称
        Returns:
            str: 缓存文件夹路径
        """

        try:
            return self.runner_dir_dict_[runner_name]["cache"]
        except KeyError:
            logger.error("runner {} not registered".format(runner_name))
            raise KeyError("runner {} not registered".format(runner_name))

    def queryLogPath(self, runner_name: str) -> str:
        """
        查询日志文件夹路径
        Args:
            runner_name (str): runner名称
        Returns:
            str: tensorboard日志文件夹路径
        """
        try:
            return self.runner_dir_dict_[runner_name]["log"]
        except KeyError:
            logger.error("runner {} not registered".format(runner_name))
            raise KeyError("runner {} not registered".format(runner_name))

    def queryDataUnitFile(self, runner_name: str) -> str:
        """
        查询数据文件路径
        Args:
            runner_name (str): runner名称
        Returns:
            str: 数据文件路径
        """
        try:
            return os.path.join(self.runner_dir_dict_[runner_name]["data_unit"], "data_unit.yaml")
        except KeyError:
            logger.error("runner {} not registered".format(runner_name))
            raise KeyError("runner {} not registered".format(runner_name))

    def queryEnvInfoFile(self, runner_name: str) -> str:

        try:
            return os.path.join(self.runner_dir_dict_[runner_name]["data_unit"], "env_info.yaml")
        except KeyError:
            logger.error("runner {} not registered".format(runner_name))
            raise KeyError("runner {} not registered".format(runner_name))

    def saveDataUnitInfo(self, runner_name: str, data_unit_status: dict):
        """
        Save data unit to data unit file.

        :param runner_name: runner name
        :type runner_name: str
        :param data_unit: data unit
        :type data_unit: dict
        """
        file_path = self.queryDataUnitFile(runner_name)

        with open(file_path, 'w') as f:
            yaml.dump(data_unit_status, f)

        logger.info("successfully save data unit to {}".format(file_path))

    def saveEnvInfo(self, runner_name: str, env_info: dict):
        """
        Save environment information to environment information file.

        :param runner_name: runner name
        :type runner_name: str
        :param env_info: environment information
        :type env_info: dict
        """
        file_path = self.queryEnvInfoFile(runner_name)

        with open(file_path, 'w') as f:
            yaml.dump(env_info, f)

    # 统一路径管理接口
    def ensureDir(self, dir_path: str) -> bool:
        """
        确保目录存在，如果不存在则创建
        
        Args:
            dir_path (str): 目录路径
            
        Returns:
            bool: True表示目录被创建，False表示目录已存在
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            return True
        else:
            logger.debug(f"Directory already exists: {dir_path}")
            return False
    
    def getCheckpointDir(self, runner_name: str) -> str:
        """
        获取检查点目录路径
        
        Args:
            runner_name (str): runner名称
            
        Returns:
            str: 检查点目录路径
        """
        checkpoint_dir = os.path.join(self.queryLogPath(runner_name), "checkpoints")
        self.ensureDir(checkpoint_dir)
        return checkpoint_dir
    
    def getCheckpointPath(self, runner_name: str, checkpoint_name: str) -> str:
        """
        获取检查点文件路径
        
        Args:
            runner_name (str): runner名称
            checkpoint_name (str): 检查点名称
            
        Returns:
            str: 检查点文件路径
        """
        checkpoint_dir = self.getCheckpointDir(runner_name)
        return os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
    
    def getModelPath(self, runner_name: str, model_name: str) -> str:
        """
        获取模型文件路径
        
        Args:
            runner_name (str): runner名称
            model_name (str): 模型名称
            
        Returns:
            str: 模型文件路径
        """
        model_dir = self.queryHistoryModelPath(runner_name)
        self.ensureDir(model_dir)
        return os.path.join(model_dir, f"{model_name}.pt")
    
    def getLogDir(self, runner_name: str) -> str:
        """
        获取日志目录路径（用于coordinator等）
        
        Args:
            runner_name (str): runner名称
            
        Returns:
            str: 日志目录路径
        """
        log_dir = self.queryLogPath(runner_name)
        self.ensureDir(log_dir)
        return log_dir
    
    def getTensorboardLogDir(self, runner_name: str) -> str:
        """
        获取TensorBoard日志目录路径
        
        Args:
            runner_name (str): runner名称
            
        Returns:
            str: TensorBoard日志目录路径
        """
        tb_dir = os.path.join(self.queryLogPath(runner_name), "tensorboard")
        self.ensureDir(tb_dir)
        return tb_dir
    
    def getExperimentDir(self, runner_name: str) -> str:
        """
        获取实验目录路径
        
        Args:
            runner_name (str): runner名称
            
        Returns:
            str: 实验目录路径
        """
        exp_dir = os.path.join(self.workspace_dir_, runner_name)
        self.ensureDir(exp_dir)
        return exp_dir
