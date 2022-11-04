from AquaML.data.DataUnit import DataUnit
from AquaML.common.ParallelComputeTool import *


class ReplayPolicy:
    def __init__(self, data: DataUnit, work_num, work_id):
        self.data = data

        start_pointer, end_pointer = ComputeDataBlock(self.data.total_length, work_num, work_id)

        self.start_pointer = start_pointer
        self.end_pointer = end_pointer

        self.pointer = start_pointer

    def reset(self):
        self.pointer = self.start_pointer

    def close(self):
        self.data.close()

    def get_action(self, obs: dict = {}):
        action = self.data.get_signal_data(self.pointer)
        self.pointer += 1

        return action
