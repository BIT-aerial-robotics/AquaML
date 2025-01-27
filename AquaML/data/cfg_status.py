'''
unit configuration以及unit状态类。
'''
import torch
from AquaML.utils import configclass
from dataclasses import MISSING

@configclass
class unitCfg:

    name: str = MISSING
    """
    必要参数，数据名称。
    该参数用于识别或者读取共享信息，每一个都具有唯一标识。
    """

    dtype:any = MISSING
    """
    必要参数，数据类型。
    """
    single_shape: tuple = MISSING
    """
    必要参数，单个数据的形状。
    """

    size: int = MISSING
    """
    必要参数，数据的长度
    """

    mode: str = "numpy"
    """
    非必要，数据的存储模式。
    """

    device:any = torch.device('cpu')
    """
    非必要，数据的设备。
    """

    shape: tuple = None
    """
    自动计算得到的数据形状。
    由single_shape和size计算得到。
    """

    bytes: int = None
    """
    数据的字节数。
    自动计算得到。
    """
