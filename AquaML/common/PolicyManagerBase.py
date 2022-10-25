import abc


class PolicyManagerBase(abc.ABC):
    """
    Each type of algo has different policy manager.
    """
    @abc.abstractmethod
    def sync(self, sync_path):
        """

        :param sync_path:
        :return:
        """