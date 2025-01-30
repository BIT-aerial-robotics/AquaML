from AquaML.utils import configclass
from dataclasses import MISSING


@configclass
class PolicyStatus:
    """
    策略状态，用于记录策略的状态信息。
    比如输入数据，输出数据等。
    """

    input_nams: list = MISSING
    """
    必要参数，输入数据的名称。
    """
