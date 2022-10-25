import abc


class AlgoBase(abc.ABC):

    def __init__(self, algo_args, data_collector, policy):
        self.algo_args = algo_args
        self.data_collector = data_collector
        self.policy = policy

    @abc.abstractmethod
    def _optimize(self, data_dict, args: dict):
        """
        run the optimize function.
        :param data_dict: When using optimization, which data will be used.
        :param args: args for optimize data.
        :return:
        """
