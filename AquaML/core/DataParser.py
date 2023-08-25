import numpy as np
import tensorflow as tf
from copy import deepcopy
from functools import partial

try:
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
except:
    pad_sequences = tf.keras.utils.pad_sequences


def pad(
        seq_start_indices: np.ndarray,
        seq_end_indices: np.ndarray,
        minimum_length: int,
        tensor: np.ndarray,
        padding_value: float = 0.0,

) -> tf.Tensor:
    """
    copy from ssb3_contrib.common.buffers
    Args:
        seq_start_indices:
        seq_end_indices:
        tensor (object):
        padding_value:

    Returns:
        tf.tensor: shape = (n_episode, seq_len, **tensor.shape[1])

    """

    seq = []

    for start, end in zip(seq_start_indices, seq_end_indices):
        if end - start < minimum_length:
            continue
        seq.append(tf.cast(tensor[start: end], dtype=tf.float32))

    # seq = [tf.cast(tensor[start: end + 1], dtype=tf.float32) for start, end in zip(seq_start_indices, seq_end_indices)]

    return pad_sequences(
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
        minimum_length: int = 1,
):
    """
    copy from ssb3_contrib.common.buffers
    Args:
        episode_starts:
        env_change:

    Returns:

    """

    seq_start = np.logical_or(episode_starts, env_change).flatten()
    seq_start[-1] = True
    seq_end_indices = np.where(seq_start == True)[0] + 1
    # seq_end_indices = np.concatenate([seq_start_indices[1:], np.array([len(episode_starts)])])
    seq_start_indices = np.concatenate([np.array([0]), seq_end_indices[:-1]])
    local_pad = partial(pad, seq_start_indices, seq_end_indices, minimum_length)

    real_start_indices = []

    for start, end in zip(seq_start_indices, seq_end_indices):
        if end - start < minimum_length:
            continue
        real_start_indices.append(start)

    # Don't pad the sequence
    return real_start_indices, local_pad


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
                 data_dict: dict or list,
                 max_size: int = 10000,
                 # for RL
                 rollout_steps: int = 1,
                 IOInfo=None,
                 num_envs: int = 1,
                 ):
        """
        Args:
            data_dict (dict): data dict.
        """

        if isinstance(data_dict, dict):
            self.data_dict = data_dict
            names = tuple(self.data_dict.keys())
            # self.batch_size = 32

            self.buffer_size = self.data_dict[names[0]].shape[0]


        elif isinstance(data_dict, list) or isinstance(data_dict, tuple):
            self.data_dict = dict()

            for name in data_dict:
                shape = IOInfo.data_info.shape_dict[name]
                if len(shape) == 2:
                    shape = (shape[1],)

                else:
                    shape = shape[1:]
                self.data_dict[name] = np.zeros((max_size, *shape))

            self.buffer_size = 0

        else:
            raise TypeError("data_dict must be dict or list")

        self.max_size = max_size

        self.current_new_index = 0

        # for RL
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs

    def random_sample(self, batch_size: int, name_list: list, tf_dataset: bool = True):

        indices = np.random.permutation(self.buffer_size)

        required_data = []

        random_indices = indices[:batch_size]

        for name in name_list:
            if tf_dataset:
                cache = tf.convert_to_tensor(self.data_dict[name][random_indices], dtype=tf.float32)
            else:
                cache = self.data_dict[name][random_indices]
            required_data.append(cache)

        return required_data
    
    def random_sample_all(self, batch_size: int, tf_dataset: bool = True)->dict:
            
            indices = np.random.permutation(self.buffer_size)
    
            return_dict = {}
            
            random_indices = indices[:batch_size]
            
            for name in self.data_dict.keys():
                
                if tf_dataset:
                    cache = tf.convert_to_tensor(self.data_dict[name][random_indices], dtype=tf.float32)
                else:
                    cache = self.data_dict[name][random_indices]
                return_dict[name] = cache
                
            return return_dict

    def get_required_data(self, name_list: list):

        required_data = []

        for name in name_list:
            required_data.append(self.data_dict[name])

        return required_data

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

    def add_data_by_buffer(self, data_dict: dict):

        # TODO: check bugs
        # check data_dict size

        sizes = []

        for key, val in data_dict.items():
            sizes.append(val.shape[0])

        mean_size = np.mean(sizes)

        if abs(mean_size - sizes[0]) > 1e-5:
            raise ValueError("data_dict size must be equal")

        mean_size = sizes[0]

        # add data

        if mean_size > self.max_size:

            for key, val in self.data_dict.items():
                self.data_dict[key][:] = data_dict[key][:self.max_size]
        else:
            remian_size = self.max_size - self.current_new_index

            cache_size = min(remian_size, mean_size)

            for key, val in self.data_dict.items():
                self.data_dict[key][self.current_new_index:self.current_new_index + cache_size] = data_dict[key][
                                                                                                  :cache_size]

                remain_cache_size = mean_size - remian_size
                # print(remain_cache_size)

                if remain_cache_size > 0:
                    self.data_dict[key][:remain_cache_size] = data_dict[key][cache_size:]

        self.current_new_index = (self.current_new_index + mean_size) % self.max_size

        self.buffer_size = min(self.buffer_size + mean_size, self.max_size)

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

                batch['bool_mask'] = tf.ones_like(batch['total_reward'], dtype=tf.bool)
                batch['bool_mask'] = np.squeeze(batch['bool_mask'])
                batch['bool_mask'] = tf.cast(batch['bool_mask'], dtype=tf.bool)

                yield batch

                start_index = end_index

        elif mode == 'seq':
            # TODO: check
            defualt_args = args = {
                'split_point_num': 1,
                'return_first_hidden': True,
                'minimum_seq_len': 16,
            }
            if args is not None:
                defualt_args.update(args)

            indices = np.arange(self.buffer_size)

            split_point_num = defualt_args['split_point_num']

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

                if l < 3:
                    indices = np.concatenate([sequence_slice[1], sequence_slice[0]])
                else:
                    sequence_slice_index = np.random.permutation(np.arange(l))

                    indices = np.concatenate([sequence_slice[i] for i in sequence_slice_index])

                # sequence_slice_index = np.random.permutation(np.arrange(l))

                # indices = np.concatenate([sequence_slice[i] for i in sequence_slice_index])

            env_change = np.zeros(self.buffer_size).reshape(self.num_envs, self.rollout_steps)

            env_change[:, self.rollout_steps - 1] = 1

            env_change = env_change.reshape(-1, 1)

            start_index = 0

            episode_start = 1 - self.data_dict['mask']

            while start_index < self.buffer_size:
                end_index = min(start_index + batch_size, self.buffer_size)
                batch_indices = indices[start_index:end_index]

                self.seq_start_indices, self.pad = create_sequencers(
                    episode_start[batch_indices],
                    env_change[batch_indices],
                    minimum_length=defualt_args['minimum_seq_len'],
                )  # padding the sequence shape (batch_size, seq_len, ...)

                batch = {}

                for key, val in self.data_dict.items():
                    if 'hidden' in key:
                        if defualt_args['return_first_hidden']:
                            batch[key] = tf.cast(val[batch_indices][self.seq_start_indices], tf.float32)
                        else:
                            batch[key] = val[batch_indices]
                    else:
                        if key == 'prob':
                            batch[key] = self.pad(val[batch_indices], 1)
                        else:
                            batch[key] = self.pad(val[batch_indices])

                batch['bool_mask'] = self.pad(np.ones_like(self.data_dict['total_reward'][batch_indices])) > 1e-6

                batch['bool_mask'] = np.squeeze(batch['bool_mask'], axis=-1)

                batch['bool_mask'] = tf.cast(batch['bool_mask'], tf.bool)

                yield batch
                start_index = end_index

        else:
            raise NotImplementedError
