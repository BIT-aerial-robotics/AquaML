import numpy as np
import tensorflow as tf
from AquaML.args.CommonArgs import DatesetArgs
import AquaML as A


class Dataset:
    """
    Now just support tf_data_set mode
    """

    def __init__(self, workspace: str, args: DatesetArgs, name=None):
        """

        :param workspace: Assume date store under workspace.
        :param name: used in MPI.
        """
        self.it = None
        self._tf_data_set = None
        self._test_data_set = None
        self.workspace = workspace
        self.args = args
        self.name = name
        self._data_set = None

    def numpy_load(self, path, share_memery=A.NO_SHARE_MEMORY):
        if share_memery == A.NO_SHARE_MEMORY:
            self._data_set = np.load(path)
        # elif share_memery == A.CREATE_SHARE_MEMORY:
        #     buff = np.load(path)
        # TODO: implement share memory

    def split_test_data(self, ratio=None):
        if ratio is None:
            ratio = self.args.test_size

        test_size = int(len(self._data_set) * ratio)

        self._test_data_set = self._data_set[-test_size:]

    def get_data_from_numpy(self, ndarray, share_memery=A.NO_SHARE_MEMORY):
        """
        Preprocess your data.
        :param ndarray:
        :param share_memery:
        :return:
        """
        # TODO: implement share memory
        self._data_set = ndarray

    def convert_tf_data_set(self):
        self._tf_data_set = tf.data.Dataset.from_tensor_slices(self._data_set).shuffle(len(self._data_set)).batch(
            self.args.batch_size)
        self._initialize_iter()

    def _initialize_iter(self):
        self.tf_it = iter(self._tf_data_set)

    def sample_index(self):
        pass

    def get_data(self):
        try:
            data = next(self.it)
        except:
            self._initialize_iter()
            return None

        return tf.cast(data, dtype=tf.float32)
