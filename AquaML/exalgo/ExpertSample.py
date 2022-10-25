from AquaML.common.AlgoBase import AlgoBase
from AquaML.data.DataCollector import DataCollector


class ExpertSample(AlgoBase):
    def _optimize(self, data_dict, args: dict):
        pass

    def __init__(self, data_collector: DataCollector):

        self.data_collector = data_collector
