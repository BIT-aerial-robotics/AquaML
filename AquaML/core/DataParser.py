import numpy as np
import tensorflow as tf
from copy import deepcopy
from functools import partial


def pad(
        seq_start_indices: np.ndarray,
        seq_end_indices: np.ndarray,
        tensor: np.ndarray,
        padding_value: float = 0.0,
) -> tf.Tensor:
    """
    copy from ssb3_contrib.common.buffers
    Args:
        seq_start_indices:
        seq_end_indices:
        tensor:
        padding_value:

    Returns:
        tf.tensor: shape = (n_episode, seq_len, **tensor.shape[1])

    """

    seq = [tf.cast(tensor[start: end + 1], dtype=tf.float32) for start, end in zip(seq_start_indices, seq_end_indices)]

    return tf.keras.utils.pad_sequences(
        seq,
        padding='post',
        dtype='float32',
        value=padding_value,
    )


# def pad_and_flatten(
#         seq_start_indices: np.ndarray,
#         seq_end_indices: np.ndarray,
#         tensor: np.ndarray,
#         padding_value: float = 0.0,
# ) -> np.ndarray:
#     return pad(seq_start_indices, seq_end_indices, tensor, padding_value).flatten()


def create_sequencers(
        episode_starts: np.ndarray,
        env_change: np.ndarray,
):
    """
    copy from ssb3_contrib.common.buffers
    Args:
        episode_starts:
        env_change:

    Returns:

    """

    seq_start = np.logical_or(episode_starts, env_change).flatten()
    seq_start[0] = True
    seq_start_indices = np.where(seq_start == True)[0]
    seq_end_indices = np.concatenate([seq_start_indices[1:], np.array([len(episode_starts)])])

    local_pad = partial(pad, seq_start_indices, seq_end_indices)

    # Don't pad the sequence
    return seq_start_indices, local_pad


class DataInfo:
    """
    Information of dateset or buffer.
    """

    def __init__(self, names: tuple or list, shapes: tuple or list, dtypes, dataset=None):
        """Data info struct.

        Args:
            names (tuple): data names.
            shapes (tuple): shapes 
            dtypes (tuple, optional): dtypes.
        """

        # TODO: 当前buffer size
        self.shape_dict = dict(zip(names, shapes))

        self.names = names

        if isinstance(dtypes, tuple) or isinstance(dtypes, list):
            self.type_dict = dict(zip(names, dtypes))
        else:
            self.type_dict = dict()
            for key in names:
                self.type_dict[key] = dtypes
        if dataset is not None:
            self.dataset_dict = dict(zip(names, dataset))
        else:
            self.dataset_dict = None

    def add_info(self, name: str, shape, dtype):
        """add info.

        Args:
            name (str): name.
            shape (tuple): shape.
            dtype (type): dtype.
        """
        self.shape_dict[name] = shape
        self.type_dict[name] = dtype

        # add element to names
        names = list(self.names)
        names.append(name)
        self.names = tuple(names)

    @property
    def get_total_size(self):
        """get batch size.
        """
        return self.shape_dict[self.names[0]][0]

    def keys(self):
        return self.names


class DataSet:

    def __init__(self,
                 data_dict: dict,

                 # for RL
                 rollout_steps: int = 1,
                 num_envs: int = 1,
                 ):
        """
        Args:
            data_dict (dict): data dict.
        """
        self.data_dict = data_dict
        names = tuple(self.data_dict.keys())
        # self.batch_size = 32

        self.buffer_size = self.data_dict[names[0]].shape[0]

        # for RL
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs

    def slice_data(self, start: int, end: int):
        """slice data.

        Args:
            start (int): start index.
            end (int): end index.
        """
        new_data_dict = {}
        for key in self.data_dict.keys():
            new_data_dict[key] = self.data_dict[key][start:end]

        return DataSet(new_data_dict)

    def __call__(self, batch_size: int, mode=None, args=None):
        """
        sample data

        Args:
            batch_size (int): batch size.

        """

        if mode is None:
            indices = np.random.permutation(self.buffer_size)

            start_index = 0

            while start_index < self.buffer_size:
                end_index = min(start_index + batch_size, self.buffer_size)
                batch_indices = indices[start_index:end_index]

                batch = {}
                for key, val in self.data_dict.items():
                    batch[key] = tf.cast(deepcopy(val[batch_indices]), tf.float32)

                batch['bool_mask'] = tf.ones_like(batch['reward'], dtype=tf.bool)

                yield batch

                start_index = end_index

        elif mode == 'seq':
            # TODO: check
            if args is None:
                args = {
                    'split_point_num': 1,
                    'return_first_hidden': True,
                    # 'shuffle_seq': True,
                }

            indices = np.arange(self.buffer_size)

            split_point_num = args['split_point_num']

            if split_point_num > 0:
                split_index = np.random.randint(0, self.buffer_size, split_point_num)
                split_index = np.sort(split_index)
                split_end_index = np.append(split_index, self.buffer_size).tolist()
                split_start_index = np.append(0, split_index).tolist()

                sequence_slice = []
                for start_index, end_index in zip(split_start_index, split_end_index):
                    sequence_slice.append(deepcopy(indices[start_index:end_index]))

                # shuffle the sequence and concat

                # sequence_slice = np.concatenate(sequence_slice)

                l = len(sequence_slice)

                if l <3:
                    indices = np.concatenate([sequence_slice[1], sequence_slice[0]])
                else:
                    sequence_slice_index = np.random.permutation(np.arange(l))

                    indices = np.concatenate([sequence_slice[i] for i in sequence_slice_index])

                # sequence_slice_index = np.random.permutation(np.arrange(l))

                # indices = np.concatenate([sequence_slice[i] for i in sequence_slice_index])

            env_change = np.zeros(self.buffer_size).reshape(self.num_envs, self.rollout_steps)

            env_change[:, 0] = 1

            env_change = env_change.reshape(-1, 1)

            start_index = 0

            episode_start = 1 - self.data_dict['mask']

            while start_index < self.buffer_size:
                end_index = min(start_index + batch_size, self.buffer_size)
                batch_indices = indices[start_index:end_index]

                self.seq_start_indices, self.pad = create_sequencers(
                    episode_start[batch_indices],
                    env_change[batch_indices],
                )  # padding the sequence shape (batch_size, seq_len, ...)

                batch = {}

                for key, val in self.data_dict.items():
                    if 'hidden' in key:
                        if args['return_first_hidden']:
                            batch[key] = tf.cast(val[batch_indices][self.seq_start_indices], tf.float32)
                        else:
                            batch[key] = val[batch_indices]
                    else:
                        if key == 'prob':
                            batch[key] = self.pad(val[batch_indices], 1)
                        else:
                            batch[key] = self.pad(val[batch_indices])

                batch['bool_mask'] = self.pad(np.ones_like(val[batch_indices])) > 1e-6

                batch['bool_mask'] = np.squeeze(batch['bool_mask'], axis=-1)

                batch['bool_mask'] = tf.cast(batch['bool_mask'], tf.bool)

                yield batch
                start_index = end_index

        else:
            raise NotImplementedError
