from AquaML.data.DataPool import DataPool
import numpy as np
import tensorflow as tf


# hierarchical_info = {'hierarchical': 0, 'start_pointer': -1, 'end_pointer': -1}


class DataManager:
    def __init__(self, obs_dic: dict, action_dic: dict, actor_input_info: list, critic_input_info: list,
                 reward_list: list, total_length: int,
                 hierarchical_info: dict = {'hierarchical': 0, 'start_pointer': -1, 'end_pointer': -1},
                 work_space=None):
        """
        It is used to manage the storage, reading, creation and management of DataPool.

        :param obs_dic: (dict) Describe observation of the environment. Support multi-input of agent。If an environment is POMDP, position is observable while velocity is hidden state, this params' input can be obs_dic = {'pos': (3,), 'vel': (3,)}.
        :param action_dic: (dict) Describe the output of actor policy. Such as gaussian policy, the action_dic can be writen as {'action':(3,),'prob':(3,)}
        :param actor_input_info: (list) Describe information of the input of actor. {'vel','image'...}
        :param critic_input_info: (list) Describe information of the input of critic.
        :param reward_list: (list) Describe every part of reward function. But the dict must contain 'total_reward' key.
        :param total_length: (int) Total length of this data.
        :param hierarchical_info: (dic) Describe multi thread information. It must contain 'hierarchical','start_pointer','end_pointer'. Default is {'hierarchical': 0, 'start_pointer': -1, 'end_pointer': -1}. 'hierarchical' describe thread level.
        :param work_space: (str) The space is stored data.
        """

        self.obs = dict()

        for key, value in obs_dic.items():
            self.obs[key] = DataPool(name=work_space + "_" + key, shape=value, total_length=total_length,
                                     hierarchical=hierarchical_info['hierarchical'])

        # self.obs[key].data

        self.next_obs = dict()
        for key, value in obs_dic.items():
            self.next_obs[key] = DataPool(name=work_space + "_next_" + key, shape=value, total_length=total_length,
                                          hierarchical=hierarchical_info['hierarchical'])

        self.action = dict()
        for key, value in action_dic.items():
            self.action[key] = DataPool(name=work_space + "_" + key, shape=value, total_length=total_length,
                                        hierarchical=hierarchical_info['hierarchical'])
        self.reward = dict()
        for key in reward_list:
            self.reward[key] = DataPool(name=work_space + "_" + key, shape=(1,), total_length=total_length,
                                        hierarchical=hierarchical_info['hierarchical'])

        self.mask = DataPool(name=work_space + '_' + 'mask', shape=(1,), total_length=total_length,
                             hierarchical=hierarchical_info['hierarchical'], dtype=np.int32)

        # store data address
        self.mapping_dict = {'obs': self.obs, 'action': self.action, 'reward': self.reward, 'next_obs': self.next_obs}

        self.actor_input_info = actor_input_info
        self.critic_input_info = critic_input_info

        self.start_pointer = hierarchical_info['start_pointer']
        self.end_pointer = hierarchical_info['end_pointer']

        self.work_space = work_space

        if self.start_pointer == -1:
            self.start_pointer = 0
            self.end_pointer = total_length - 1

        self.pointer = self.start_pointer

    def store(self, obs: dict, action: dict, reward: dict, next_obs: dict, mask: int):
        """
        Store data in data_pool.

        :param obs: (dict) observation.
        :param action: (dict) action.
        :param reward: (dict) reward.
        :param next_obs: (dict) next_obs.
        :param mask: (int) The episode end flag. When mask=1, episode is end.
        :return: None.
        """
        index = self.pointer

        # store observation
        for key, value in obs.items():
            self.obs[key].store(value, index)

        # store action
        for key, value in action.items():
            self.action[key].store(value, index)

        # store reward
        for key, value in reward.items():
            self.reward[key].store(value, index)

        # store next_obs
        for key, value in next_obs.items():
            self.next_obs[key].store(value, index)

        self.mask.store(mask, index)

        self.pointer = self.pointer + 1

    def reset(self):
        """
        Reset pointer.

        :return:
        """
        self.pointer = self.start_pointer

    def close(self):
        """
        Release memory.

        :return: None.
        """
        for key, value in self.obs.items():
            value.close()

        for key, value in self.action.items():
            value.close()

        for key, value in self.reward.items():
            value.close()

    def slice_data(self, name: list, start: int, end: int):
        """
        Slice the selected data.

        :param name: (list) The data you want to select.
        :param start: (int)
        :param end: (int)
        :return: dict, if input is {'obs','action'},
                       return will be like:
                                    {'obs':{'vel':value,...},'action':{'action':value,'prob': value}           }
        """
        ret_dic = dict()
        for key1 in name:
            data_dict = self.mapping_dict[key1]
            sub_dict = dict()
            for key2, values in data_dict.items():
                value_ = values.data[start:end]
                sub_dict[key2] = value_
            ret_dic[key1] = sub_dict

        return ret_dic

    def get_input_data(self, actor_is_batch_timesteps=False, critic_is_batch_timesteps=False,
                       args_dict: dict = {'tf_data': True}):
        """
        Return like {'actor':[data1,data2,...],'next_actor':[data1,data2,...]}

        :return: dict contains tensor.
        """
        actor_inputs_data = []

        for key1 in self.actor_input_info:
            actor_inputs_data.append(self.obs[key1].data)

        next_actor_input_data = []

        for key1 in self.actor_input_info:
            next_actor_input_data.append(self.next_obs[key1].data)

        critic_inputs_data = []

        for key1 in self.critic_input_info:
            critic_inputs_data.append(self.obs[key1].data)

        next_critic_inputs_data = []

        for key1 in self.critic_input_info:
            next_critic_inputs_data.append(self.next_obs[key1].data)

        act = []

        for key1, values in self.action.items():
            act.append(tf.cast(values.data, dtype=tf.float32))

        out_dict = {'actor': actor_inputs_data, 'next_actor': next_actor_input_data, 'critic': critic_inputs_data,
                    'next_critic': next_critic_inputs_data, 'action': act}
        mapping_dict = {
            'actor': list(self.actor_input_info),
            'next_actor': list(self.actor_input_info),
            'critic': list(self.critic_input_info),
            'next_critic': list(self.critic_input_info),
            'action': list(self.action)
        }
        if actor_is_batch_timesteps:
            buffer_actor = out_dict['actor']
            buffer = dict(zip(mapping_dict['actor'], buffer_actor))
            buffer = self.batch_timesteps(buffer, args_dict.get('traj_length'), args_dict.get('overlap_size'))
            out_dict['actor'] = list(buffer.values())

            buffer_actor = out_dict['next_actor']
            buffer = dict(zip(mapping_dict['next_actor'], buffer_actor))
            buffer = self.batch_timesteps(buffer, args_dict.get('traj_length'), args_dict.get('overlap_size'))
            out_dict['next_actor'] = list(buffer.values())

            buffer_critic = out_dict['critic']
            buffer = dict(zip(mapping_dict['critic'], buffer_critic))
            buffer = self.batch_timesteps(buffer, args_dict.get('traj_length'), args_dict.get('overlap_size'))
            out_dict['critic'] = list(buffer.values())

            buffer_critic = out_dict['next_critic']
            buffer = dict(zip(mapping_dict['next_critic'], buffer_critic))
            buffer = self.batch_timesteps(buffer, args_dict.get('traj_length'), args_dict.get('overlap_size'))
            out_dict['next_critic'] = list(buffer.values())

            buffer_action = out_dict['action']
            buffer = dict(zip(mapping_dict['action'], buffer_action))
            buffer = self.batch_timesteps(buffer, args_dict.get('traj_length'), args_dict.get('overlap_size'))
            out_dict['action'] = list(buffer.values())

        if critic_is_batch_timesteps:
            buffer_critic = out_dict['critic']
            buffer = dict(zip(mapping_dict['critic'], buffer_critic))
            buffer = self.batch_timesteps(buffer, args_dict.get('traj_length'), args_dict.get('overlap_size'))
            out_dict['critic'] = list(buffer.values())

            buffer_critic = out_dict['next_critic']
            buffer = dict(zip(mapping_dict['next_critic'], buffer_critic))
            buffer = self.batch_timesteps(buffer, args_dict.get('traj_length'), args_dict.get('overlap_size'))
            out_dict['next_critic'] = list(buffer.values())

        if args_dict['tf_data']:
            for key, value in out_dict.items():
                buffer = dict(zip(mapping_dict[key], value))
                tf_data = self.convert_tensor(buffer)
                out_dict[key] = list(tf_data.values())

        return out_dict

    def batch_timesteps(self, input_dic: dict, traj_length=None, overlap_size=None):
        """
        Convert input data into time step (batchsize,timestep,features). This function is used in RNN.

        :param input_dic: (dict) The data is needed to convert.
        :param overlap_size:
        :param traj_length: overlap_size and traj_length must be used at the same time.
        :return:
        """
        index_done = np.where(self.mask.data == 0)[0] + 1

        start_index = 0
        move_step = None

        if traj_length is not None:
            if overlap_size is not None:
                move_step = traj_length - overlap_size
            else:
                move_step = traj_length

        output_dic = dict()

        for key in list(input_dic):
            output_dic[key] = []

        for end_index in index_done:
            buffer = dict()

            for key, value in input_dic.items():
                buffer[key] = value[start_index:end_index]

            if traj_length is not None:
                start_ind = 0
                end_ind = int(start_index + traj_length)
                while end_ind <= end_index:
                    for key, value in buffer.items():
                        output_dic[key].append(np.expand_dims(value[start_ind:end_ind]))

                    start_ind = start_ind + move_step
                    end_ind = start_ind + traj_length
            else:
                for key, value in buffer.items():
                    output_dic[key].append(np.expand_dims(value, axis=0))

            start_index = end_index

        for key, value in output_dic.items():
            output_dic[key] = np.vstack(value)
        return output_dic

    @staticmethod
    def slice_tuple_list(data, start, end):
        buffer = []

        for array in data:
            buffer.append(array[start: end])

        return buffer

    @staticmethod
    def convert_tensor(input_dic: dict):
        """

        :param input_dic: {'name':array}
        :return: dict
        """
        buffer = dict()
        for key, value in input_dic.items():
            buffer[key] = tf.cast(value, tf.float32)

        return buffer

    @staticmethod
    def batch_features(input_tup: list, convert_tensor=False):
        """

        :param input_tup: (data1,data2,..)
        :return:
        """
        length = len(input_tup)
        name = {str(i) for i in range(length)}

        input_dic = dict(zip(name, input_tup))

        out_dic = dict()

        for key in list(input_dic):
            out_dic[key] = []

        for key, values in input_dic.items():
            for value in values:
                out_dic[key].append(value)

        for key, value in out_dic.items():
            out_dic[key] = np.vstack(value)

        if convert_tensor:
            out_dic = DataManager.convert_tensor(out_dic)

        return list(out_dic.values())


if __name__ == "__main__":
    obs_dic = {'pos': (3,), 'vel': (3,), 'omega': (3,)}

    data_pool_dic = dict()

    for key, value in obs_dic.items():
        data_pool_dic[key] = DataPool(key, value, 10, dtype=np.int32)

    for key, data_pool in data_pool_dic.items():
        print(data_pool.shape)
