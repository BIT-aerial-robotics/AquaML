import torch  # 引用 PyTorch 包
from AquaML.torch.TorchAlgoBase import TorchRLAlgoBase
from AquaML.param.ParamBase import RLParmBase
from AquaML import settings
import numpy as np
from AquaML.torch.Dataset import RLDataSet

# 定义 SACParam 类继承自 RLParmBase 用于设置 SAC 算法的超参数
class SACParam(RLParmBase):
    def __init__(self, 
                 total_timesteps: int, 
                 buffer_size: int, 
                 gamma: float, 
                 tau: float, 
                 batch_size: int, 
                 learning_starts: int, 
                 policy_lr: float, 
                 q_lr: float, 
                 policy_frequency: int, 
                 target_network_frequency: int, 
                 alpha: float, 
                 autotune: bool,
                 env_num: int=1,
                 max_step: int=np.inf,
                 envs_args: dict={},
                 checkpoints_store_interval: int = 5,
                 independent_model: bool = True,
                ):
        """
        SAC 算法的超参数。

        Args:
            total_timesteps (int): 总时间步。
            buffer_size (int): 缓冲区大小。
            gamma (float): 折扣因子。
            tau (float): 目标平滑系数。
            batch_size (int): 批大小。
            learning_starts (int): 开始学习的时间步。
            policy_lr (float): 策略网络的学习率。
            q_lr (float): Q 网络的学习率。
            policy_frequency (int): 策略训练的频率。
            target_network_frequency (int): 目标网络更新的频率。
            alpha (float): 熵正则化系数。
            autotune (bool): 是否自动调整熵系数。
            env_num (int, optional): 每个进程环境的数量. 默认为 1.
            max_step (int, optional): 最大步数. 默认为 np.inf.
            envs_args (dict, optional): 环境的参数. 默认为{}.
            checkpoints_store_interval (int, optional): 检查点存储间隔. 默认为 5.
            independent_model (bool, optional): 确认每个进程中是否有独立的模型。当为 True 时，每个进程中都有独立的模型。当为 False 时，所有进程共享一个模型. 默认为 True.
        """
        super().__init__(
            total_timesteps=total_timesteps,
            gamma=gamma,
            env_num=env_num,
            max_step=max_step,
            envs_args=envs_args,
            independent_model=independent_model,
            summary_steps=total_timesteps,
            algo_name='SAC',
            checkpoints_store_interval=checkpoints_store_interval,
        )

        self.buffer_size = buffer_size
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.alpha = alpha
        self.autotune = autotune

# 定义 SACAlgo 类继承自 TorchRLAlgoBase 实现 SAC 算法
class SACAlgo(TorchRLAlgoBase):
    def __init__(self, hyper_params, model_dict):
        """
        SAC 算法。

        Args:
            hyper_params (SACParam): SAC 算法的超参数。
            model_dict (dict): 模型字典。
        """
        super().__init__(hyper_params, model_dict)

        self._algo_name = 'SAC'  # 算法的名称，必须赋值。

        self.actor = model_dict['actor']().to(settings.device)  # 创建 actor 模型
        self.qf1 = model_dict['qf1']().to(settings.device)  # 创建 qf1 模型
        self.qf2 = model_dict['qf2']().to(settings.device)  # 创建 qf2 模型
        self.qf1_target = model_dict['qf1']().to(settings.device)  # 创建 qf1 目标模型
        self.qf2_target = model_dict['qf2']().to(settings.device)  # 创建 qf2 目标模型

        ##############################
        # 1. 初始化数据集
        ##############################

        # 将模型添加到模型字典 _model_dict 中
        self._model_dict['actor'] = self.actor
        self._model_dict['qf1'] = self.qf1
        self._model_dict['qf2'] = self.qf2
        self._model_dict['qf1_target'] = self.qf1_target
        self._model_dict['qf2_target'] = self.qf2_target

        # 创建优化器
        self.q_optimizer, self.q_optimizer_step_fn = self.create_optimizer([self.qf1, self.qf2], lr=hyper_params.q_lr)
        self.actor_optimizer, self.actor_optimizer_step_fn = self.create_optimizer(self.actor, lr=hyper_params.policy_lr)

        # 自动熵调整相关
        if hyper_params.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.actor.output_info.last_shape_dict['action']).to(settings.device)).item()
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=settings.device))
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer, self.a_optimizer_step_fn = self.create_optimizer([self.log_alpha], lr=hyper_params.q_lr)
        else:
            self.alpha = hyper_params.alpha

        # 初始化回放缓冲区
        self.rb = RLDataSet(
            capacity=hyper_params.buffer_size,
            default_type='tensor',
            default_device=settings.device
        )

        self.global_step = 0

    def _train_action(self, state):
        """
        该函数为必须实现的接口，用于训练动作，返回动作和相关信息，是进行 rollout 的函数，其输出必须包含 action 和 mu。

        Args:
            state (dict): 状态数据。
        Returns:
            actor_out (dict): actor 模型的输出。
            mu (torch.Tensor): 动作的均值。
        """
        if self.global_step < self._hyper_params.learning_starts:
            action = torch.from_numpy(np.array([self.actor.output_info.last_shape_dict['action'].sample() for _ in range(settings.env_num)])).to(settings.device)
        else:
            action, log_pi, _ = self.actor.get_action(torch.Tensor(state).to(settings.device))
        return {'action': action, 'log_pi': log_pi}, action

    def train_qf(self, data_dict):
        """
        训练 Q 函数。

        Args:
            data_dict (dict): 包含训练数据的字典。

        Returns:
            loss_dict (dict): 包含 Q 函数损失的字典。
        """
        data_set = RLDataSet(
            data_dict=data_dict,
            default_type='tensor',
            default_device=settings.device
        )

        data = data_set.get_batch(self._hyper_params.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self._hyper_params.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = torch.nn.functional.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = torch.nn.functional.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # 优化 Q 函数
        self.q_optimizer_step_fn(qf_loss)

        loss_dict = {
            'qf1_loss': qf1_loss.detach(),
            'qf2_loss': qf2_loss.detach(),
            'qf_loss': qf_loss.detach()
        }
        return loss_dict

    def train_actor(self, data_dict):
        """
        训练 actor 网络。

        Args:
            data_dict (dict): 包含训练数据的字典。

        Returns:
            loss_dict (dict): 包含 actor 损失的字典。
        """
        data_set = RLDataSet(
            data_dict=data_dict,
            default_type='tensor',
            default_device=settings.device
        )

        data = data_set.get_batch(self._hyper_params.batch_size)

        for _ in range(self._hyper_params.policy_frequency):
            pi, log_pi, _ = self.actor.get_action(data.observations)
            qf1_pi = self.qf1(data.observations, pi)
            qf2_pi = self.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            # 优化 actor
            self.actor_optimizer_step_fn(actor_loss)

        loss_dict = {
            'actor_loss': actor_loss.detach()
        }

        if self._hyper_params.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.get_action(data.observations)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            # 优化 alpha
            self.a_optimizer_step_fn(alpha_loss)

            loss_dict['alpha_loss'] = alpha_loss.detach()

        return loss_dict

    def update_target_networks(self):
        """
        更新目标网络。
        """
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self._hyper_params.tau * param.data + (1 - self._hyper_params.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self._hyper_params.tau * param.data + (1 - self._hyper_params.tau) * target_param.data)

    def train(self, data_dict):
        """
        训练函数。

        Args:
            data_dict (dict): 从环境中采集的数据。

        Returns:
            loss_tracker (object): 包含损失信息的追踪器。
        """
        # 将数据添加到回放缓冲区
        self.rb.add(data_dict)

        if self.global_step > self._hyper_params.learning_starts:
            # 从缓冲区采样数据进行训练
            data = self.rb.get_batch(self._hyper_params.batch_size)

            # 训练 Q 函数
            qf_loss_dict = self.train_qf(data)

            # 按照策略频率训练 actor
            if self.global_step % self._hyper_params.policy_frequency == 0:
                actor_loss_dict = self.train_actor(data)

            # 更新目标网络
            if self.global_step % self._hyper_params.target_network_frequency == 0:
                self.update_target_networks()

            self.global_step += 1

        return self.loss_tracker