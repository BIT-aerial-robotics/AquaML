import abc


class AlgoBase(abc.ABC):

    @abc.abstractmethod
    def _optimize(self, data_dict, args: dict):
        """
        run the optimize function.
        :param data_dict: When using optimization, which data will be used.
        :param args: args for optimize data.
        :return:
        """
