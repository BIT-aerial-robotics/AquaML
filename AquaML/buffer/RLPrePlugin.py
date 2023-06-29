"""

这里是强化学习的预处理插件，预处理首先会根据episode的长度进行截断，
然后将episode的数据进行整理，分成episode时候可以写入一些插件，比如对
episode的数据进行统计，统计reward，过滤部分episode等等。
"""
from AquaML.core.RLToolKit import RLStandardDataSet

class SplitTrajectory:

    def __init__(self):
        pass

    def __call__(self, trajectory: RLStandardDataSet):
        
        mask = trajectory.mask

    def add_plugin(self, plugin):
        pass

