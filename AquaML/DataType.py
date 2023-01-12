import numpy as np


class DataInfo:
    """
    Information of dateset or buffer.
    """
    def __init__(self, names:tuple, shapes:tuple, dtypes, dataset=None):
        """Data info srtuct.

        Args:
            names (tuple): data names.
            shapes (tuple): shapes 
            dtypes (tuple, optional): dtypes.
        """
        
        # TODO: 当前buffer size
        self.shape_dict = dict(zip(names, shapes))

        self.names = names

        if isinstance(dtypes, tuple):
            self.type_dict = dict(zip(names, dtypes))
        else:
            self.type_dict = dict()
            for key in names:
                self.type_dict[key] = dtypes
        if dataset is not None:
            self.dataset_dict = dict(zip(names, dataset))
        else:
            self.dataset_dict = None

class RLIOInfo:
    """
    Information of reinforcement learning model input and output.
    """
    def __init__(self, obs_info:dict,obs_type_info, actor_input_info:tuple, actor_out_info:dict, critic_input_info:tuple , reward_info:tuple, buffer_size:int):
        """Reinforcement learning model input-output(IO) information.

        Args:
            obs_info (dict): observation information. (name,shape)
            obs_type_info: observation type information. dict or type. Like: {'obs':np.float32} or np.float32
            actor_input_info (tuple): actor input information.
            actor_out_info (dict): actor output information. (name,shape)
            critic_input_info (tuple): critic input information.
            reward_info (tuple): reward information.
            buffer_size (int): buffer size.
        """

        # incert buffer size into shapes
        def insert_buffer_size(shape):
            shapes = []
            shapes.append(buffer_size)

            if isinstance(shape, tuple):
                for val in shape:
                    shapes.append(val)
            else:
                shapes.append(shape)
            
            shapes = tuple(shapes)

            return shapes


        # convert obs_info, actor_out_info into data_info
        # create data_info

        data_info_dict = dict()
        data_type_info_dict = dict()

        # add obs_info to data_info
        for key, shape in obs_info.items():
            data_info_dict[key] = insert_buffer_size(shape)
            if isinstance(obs_type_info, dict):
                data_type_info_dict[key] = obs_type_info[key]
            else:
                data_type_info_dict[key] = obs_type_info

        # add actor_out_info to data_info
        for key, shape in actor_out_info.items():
            data_info_dict[key] = insert_buffer_size(shape)
            data_type_info_dict[key] = np.float32

        # add reward_info to data_info
        for key in reward_info:
            data_info_dict[key] = (buffer_size, 1)
            data_type_info_dict[key] = np.float32
        
        # create data_info
        self.data_info = DataInfo(tuple(data_info_dict.keys()), tuple(data_info_dict.values()), tuple(data_type_info_dict.values()))

        self.actor_input_info = actor_input_info
        self.critic_input_info = critic_input_info

# test
if __name__ == "__main__":
    test = RLIOInfo({'obs':(1,2)}, np.float32, ('obs',), {'actor_out':(1,2)}, ('obs','actor_out'), ('reward',), 4)

    print(test.data_info.shape_dict)
    print(test.data_info.type_dict)
    print(test.actor_input_info)
    print(test.critic_input_info)

    # for name, val in test.data_info.type_dict.items():
    #     print(name, val)