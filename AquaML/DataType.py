import numpy as np

# TODO: 添加diplay()函数方便以后debug
class DataInfo:
    """
    Information of dateset or buffer.
    """
    def __init__(self, names:tuple, shapes:tuple, dtypes, dataset=None):
        """Data info struct.

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
    
    def add_info(self, name:str, shape, dtype):
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
        
# TODO:有些成员转换成私有和保护类型
# TODO: need to check before run
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
            
            actor_out_info (dict): actor output information. (name,shape) Example: {'action':(2,), 'log_std':(2,)}
            Notice: actor_out_info must have key 'action'.
            
            critic_input_info (tuple): critic input information.
            
            reward_info (tuple): reward information.
            
            buffer_size (int): buffer size.
        return:
            raise Exception: obs_info and actor_out_info must have same keys.
        """

        # insert buffer size into shapes
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
        
        # add next_obs_info to data_info
        for key in obs_info.keys(): 
            data_info_dict['next_'+key] = shape
            data_type_info_dict['next_'+key] = data_type_info_dict[key]

        # check 'action' whether in actor_out_info
        # if not, rasing error
        if 'action' not in actor_out_info:
            raise ValueError("actor_out_info must have 'action'")
        
        # add mask_info to data_info
        data_info_dict['mask'] = (buffer_size, 1)
        data_type_info_dict['mask'] = np.int32

        # add actor_out_info to data_info
        for key, shape in actor_out_info.items():
            data_info_dict[key] = insert_buffer_size(shape)
            data_type_info_dict[key] = np.float32
        
        # NOTE: actor_out contains exploration policy output
        # if 'prob' in actor_out_info, add it to data_info
        if 'prob' not in actor_out_info:
            shape = actor_out_info['action']
            data_info_dict['prob'] = insert_buffer_size(shape)
            data_type_info_dict['prob'] = np.float32

        # add reward_info to data_info
        for key in reward_info:
            data_info_dict[key] = (buffer_size, 1)
            data_type_info_dict[key] = np.float32

        # check reward info whether have 'total reward'
        # if not, add it
        if 'total_reward' not in reward_info:
            data_info_dict['total_reward'] = (buffer_size, 1)
            data_type_info_dict['total_reward'] = np.float32
        
        # create data_info
        self.data_info = DataInfo(tuple(data_info_dict.keys()), tuple(data_info_dict.values()), tuple(data_type_info_dict.values()))

        self.actor_input_info = actor_input_info # tuple
        self.critic_input_info = critic_input_info # tuple
        self.reward_info = reward_info # tuple

        # store action info
        self.actor_out_info = actor_out_info # dict
        self.actor_out_name = tuple(actor_out_info.keys()) # tuple
        self.actor_model_out_name = tuple(actor_out_info) # tuple
        
        # if 'prob' not in actor_out_info, add it
        if 'prob' not in self.actor_out_info:
            self.actor_out_info['prob'] = actor_out_info['action']
            self.actor_out_name = tuple(self.actor_out_info.keys())

        # verify exploration info
        if 'log_std' in self.actor_out_info.keys():
            self.explore_info = 'auxiliary'
        else:
            self.explore_info = 'self'
        
        # buffer size
        self.__buffer_size = buffer_size
    
    def add_info(self, name:str, shape, dtype):
        """Add information to data_info.

        Args:
            name (str): name of information.
            shape (tuple): shape of information.
            dtype (type): type of information.
        """
        self.data_info.add_info(name, shape, dtype)
        
    @property
    def buffer_size(self):
        return self.__buffer_size

# test
if __name__ == "__main__":
    test = RLIOInfo({'obs':1}, np.float32, ('obs',), {'action':(1,2)}, ('obs','action'), ('reward',), 4)
    
    test.add_info('test', (4,5), np.float32)

    print(test.data_info.shape_dict)
    print(test.data_info.type_dict)
    print(test.actor_input_info)
    print(test.critic_input_info)
    print(test.actor_out_info)
    print(test.actor_out_name)
    print(test.actor_model_out_name)

    # for name, val in test.data_info.type_dict.items():
    #     print(name, val)