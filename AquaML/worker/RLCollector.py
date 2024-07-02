import abc
import numpy as np
from AquaML import data_module, settings,recorder,logger

if settings.engine == 'torch':
    import torch


class RLCollectorBase(abc.ABC):
    """
    用于RLWorker收集数据使用,
    同时还用于承担统计数据的功能。
    """

    def __init__(self,
                 algo_name: str,
                 obs_names: list,
                 actor_output_names: list,
                 reward_names: list,
                 data_type: str = 'list',
                 summary_steps: int = 1000,
                 ):
        """
        初始化RLCollector。

        Args:
            algo_name (str): 算法名称。
            obs_names (list): observation的名称。
            action_names (list): action的名称。
            reward_names (list): reward的名称。['reward']用于最终的奖励函数，其他均看作辅助奖励。
            data_type (str): 数据类型，目前支持list和unit两种。
            summary_steps (int): 每隔多少步进行一次统计。
        """

        ############################
        # 1. 初始化参数
        ############################
        self.algo_name = algo_name
        self.obs_names = obs_names
        self.action_names = actor_output_names
        self.reward_names = reward_names
        self.summary_steps = summary_steps

        # if data_type == 'list':
        #     self.reset = self.reset_list
        # elif data_type == 'unit':
        #     self.reset = self.reset_unit

        ############################
        # 2. 添加功能性数据
        ############################

        self.all_names = (*obs_names, *actor_output_names, *reward_names, 'terminal', 'truncated')

        # 将数据加入dict中，方便后续查找。

        self.obs_dict = {}
        for obs_name in obs_names:
            self.obs_dict[obs_name] = data_module.query_data(
                name=obs_name,
                set_name=self.algo_name
            )[0]

        self.next_obs_dict = {}
        for obs_name in obs_names:
            self.next_obs_dict['next_' + obs_name] = data_module.query_data(
                name='next_' + obs_name,
                set_name=self.algo_name
            )[0]

        self.action_dict = {}
        for action_name in actor_output_names:
            self.action_dict[action_name] = data_module.query_data(
                name=action_name,
                set_name=self.algo_name
            )[0]

        self.reward_dict = {}
        for reward_name in reward_names:
            self.reward_dict[reward_name] = data_module.query_data(
                name=reward_name,
                set_name=self.algo_name
            )[0]

        self.terminal = data_module.query_data(
            name='terminal',
            set_name=self.algo_name
        )[0]

        self.truncated = data_module.query_data(
            name='truncated',
            set_name=self.algo_name
        )[0]

        ############################
        # 3. 配置summary数据
        # 每个数据的格式(num_env, 1, 1)
        # 在计算时会为为每一个环境使用(num_env, 1, 1)的格式。
        ############################

        self.summary_count = 0  # 判断是否需要进行summary

        self.summary_data = {}
        env_num = settings.env_num
        # steps = settings.steps

        for name in self.reward_names:
            # TODO：这个地方确定一下shape
            self.summary_data[name] = np.zeros(shape=(env_num, 1))

    def reset_list(self):
        """
        重置list数据。
        """

        for name in self.all_names:
            data_module.query_data_list(
                list_name=name,
                lists_name=self.algo_name
            ).reset()

    def reset_unit(self):
        """
        重置unit数据。
        """

        for name in self.all_names:
            data_module.query_data_unit(
                unit_name=name,
                units_name=self.algo_name
            ).reset()

    def reset_reward(self):
        """
        重置reward数据。
        """

        for name in self.reward_names:
            self.summary_data[name] = np.zeros(shape=(settings.env_num, 1))

        self.summary_count = 0

    def reset(self):
        """
        重置数据。
        """
        # 重制obs和next_obs
        for obs_name in self.obs_names:
            self.obs_dict[obs_name].reset()
            self.next_obs_dict['next_' + obs_name].reset()

        # 重制action
        for action_name in self.action_names:
            self.action_dict[action_name].reset()

        # 重制reward
        for reward_name in self.reward_names:
            self.reward_dict[reward_name].reset()

        # 重制terminal
        self.terminal.reset()

        # 重制truncated
        self.truncated.reset()

    def get_data(self):
        """
        获取数据所有数据，并以字典形式返回。
        """

        ret_dict = {}

        for obs_name in self.obs_names:
            ret_dict[obs_name] = np.concatenate(self.obs_dict[obs_name].get_data(), axis=1)
            ret_dict['next_' + obs_name] = np.concatenate(self.next_obs_dict['next_' + obs_name].get_data(), axis=1)

        for action_name in self.action_names:
            ret_dict[action_name] = np.concatenate(self.action_dict[action_name].get_data(), axis=1)

        for reward_name in self.reward_names:
            ret_dict[reward_name] = np.concatenate(self.reward_dict[reward_name].get_data(), axis=1)

        ret_dict['terminal'] = np.concatenate(self.terminal.get_data(), axis=1)
        ret_dict['truncated'] = np.concatenate(self.truncated.get_data(), axis=1)

        return ret_dict

    # TODO：未来升级
    @abc.abstractmethod
    def append(self,
               obs: dict,
               next_obs: dict,
               action: dict,
               reward: dict,
               terminal: np.ndarray,
               truncated: np.ndarray
               ):
        """
        将数据添加到collector中。

        Args:
            obs (dict): 当前时刻的观察值。
            next_obs (dict): 下一时刻的观察值。
            action (dict): 当前时刻的动作。
            reward (dict): 当前时刻的奖励。
            terminal (np.ndarray): 当前时刻是否终止。
            truncated (np.ndarray): 当前时刻是否截断。
        return:
            summary_flag (bool): 是否需要进行summary。
        """


class RLCollector(RLCollectorBase):
    """
    用于RLWorker收集数据使用。
    """

    def __init__(self,
                 algo_name: str,
                 obs_names: list,
                 actor_output_names: list,
                 reward_names: list,
                 summary_steps: int = 1000
                 ):
        """
        初始化RLCollector。

        Args:
            algo_name (str): 算法名称。
            obs_names (list): observation的名称。
            action_names (list): action的名称。
            reward_names (list): reward的名称。
        """

        super(RLCollector, self).__init__(
            algo_name=algo_name,
            obs_names=obs_names,
            actor_output_names=actor_output_names,
            reward_names=reward_names,
            data_type='list',
            summary_steps=summary_steps
        )

    def append(self,
               obs: dict,
               next_obs: dict,
               action: dict,
               reward: dict,
               terminal: np.ndarray,
               truncated: np.ndarray
               ):
        self.summary_count += 1

        reset_flag = False
        for obs_name in self.obs_names:
            self.obs_dict[obs_name].append(np.expand_dims(obs[obs_name], axis=1))

        for obs_name in self.obs_names:
            self.next_obs_dict['next_' + obs_name].append(np.expand_dims(next_obs[obs_name], axis=1))

        for action_name in self.action_names:
            self.action_dict[action_name].append(np.expand_dims(action[action_name], axis=1))

        self.terminal.append(np.expand_dims(terminal, axis=1))
        self.truncated.append(np.expand_dims(truncated, axis=1))

        for reward_name in self.reward_names:
            self.reward_dict[reward_name].append(np.expand_dims(reward[reward_name], axis=1))

            self.summary_data[reward_name] += reward[reward_name]

            if self.summary_count % self.summary_steps == 0:
                # TODO: 这个地方需要修改
                # self.summary_data[reward_name] += reward[reward_name]
                # 将数据推送data_module中
                data_module.rl_dict['max_' + reward_name] = np.max(self.summary_data[reward_name])
                data_module.rl_dict['min_' + reward_name] = np.min(self.summary_data[reward_name])
                data_module.rl_dict['mean_' + reward_name] = np.mean(self.summary_data[reward_name])
                
                

                reset_flag = True

                # 重置reward数据
                # self.reset_reward()

        if reset_flag:
            self.reset_reward()
            
        return reset_flag
    
class RLIsaacCollector(RLCollector):
        def __init__(self,
                    algo_name: str,
                    obs_names: list,
                    actor_output_names: list,
                    reward_names: list,
                    summary_steps: int = 1000
                    ):
            """
            初始化RLCollector。

            Args:
                algo_name (str): 算法名称。
                obs_names (list): observation的名称。
                action_names (list): action的名称。
                reward_names (list): reward的名称。
            """

            super(RLIsaacCollector, self).__init__(
                algo_name=algo_name,
                obs_names=obs_names,
                actor_output_names=actor_output_names,
                reward_names=reward_names,
                summary_steps=summary_steps
            )
            
            for name in self.reward_names:
                # TODO：这个地方确定一下shape
                self.summary_data[name] = torch.zeros(size=(settings.env_num, 1),dtype=torch.float32,device=settings.device)
                
        def reset_reward(self):
            """
            重置reward数据。
            """

            for name in self.reward_names:
                self.summary_data[name] = torch.zeros(size=(settings.env_num, 1),dtype=torch.float32,device=settings.device)
                
            self.summary_count = 0
            
        def append(self,
               obs: dict,
               next_obs: dict,
               action: dict,
               reward: dict,
               terminal: torch.Tensor,
               truncated: torch.Tensor
               ):
            self.summary_count += 1

            reset_flag = False
            for obs_name in self.obs_names:
                self.obs_dict[obs_name].append(torch.unsqueeze(obs[obs_name], axis=1))

            for obs_name in self.obs_names:
                self.next_obs_dict['next_' + obs_name].append(torch.unsqueeze(next_obs[obs_name], axis=1))

            for action_name in self.action_names:
                self.action_dict[action_name].append(torch.unsqueeze(action[action_name], axis=1))

            self.terminal.append(torch.unsqueeze(terminal, axis=1))
            self.truncated.append(torch.unsqueeze(truncated, axis=1))

            for reward_name in self.reward_names:
                self.reward_dict[reward_name].append(torch.unsqueeze(reward[reward_name], axis=1))

                self.summary_data[reward_name] += reward[reward_name]

                if self.summary_count % self.summary_steps == 0:
                    # TODO: 这个地方需要修改
                    # self.summary_data[reward_name] += reward[reward_name]
                    # 将数据推送data_module中
                    data_module.rl_dict['max_' + reward_name] = self.summary_data[reward_name].max().to('cpu').numpy()
                    data_module.rl_dict['min_' + reward_name] = self.summary_data[reward_name].min().to('cpu').numpy()
                    data_module.rl_dict['mean_' + reward_name] = self.summary_data[reward_name].mean().to('cpu').numpy()
                    
                    

                    reset_flag = True

                    # 重置reward数据
                    # self.reset_reward()

            if reset_flag:
                self.reset_reward()
                
            return reset_flag
        
        def get_data(self):
            """
            获取数据所有数据，并以字典形式返回。
            """

            ret_dict = {}

            for obs_name in self.obs_names:
                ret_dict[obs_name] = torch.cat(self.obs_dict[obs_name].get_data(), axis=1)
                ret_dict['next_' + obs_name] = torch.cat(self.next_obs_dict['next_' + obs_name].get_data(), axis=1)

            for action_name in self.action_names:
                ret_dict[action_name] = torch.cat(self.action_dict[action_name].get_data(), axis=1)

            for reward_name in self.reward_names:
                ret_dict[reward_name] = torch.cat(self.reward_dict[reward_name].get_data(), axis=1)

            ret_dict['terminal'] = torch.cat(self.terminal.get_data(), axis=1)
            ret_dict['truncated'] = torch.cat(self.truncated.get_data(), axis=1)

            return ret_dict