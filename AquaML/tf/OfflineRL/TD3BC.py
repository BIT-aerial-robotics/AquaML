import tensorflow as tf
from AquaML.algo_base.OfflineRLBase import OfflineRLBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import FileSystemBase
from AquaML.core.Tool import LossTracker
from AquaML.param.OfflineRL import TD3BCParams
from AquaML.buffer.BufferBase import BufferBase


class TD3BC(OfflineRLBase):
    """
    TD3BC算法。

    TD3BC算法是TD3算法的改进版，主要改进了actor的训练方式。
    """

    def __init__(self,
                 actor,
                 q_critic,
                 hyper_params:TD3BCParams,
                 weight_path: str,
                 # buffer,
                 communicator: CommunicatorBase,
                 data_module: DataModule,
                 file_system: FileSystemBase,
                 name: str='TD3BC'
                 ):
        """
        初始化TD3BC算法。

        Args:
            actor: actor网络模型，这里是一个class
            q_critic: q_critic网络模型，这里是一个class。
            hyper_params: 超参数。
            name (str): 算法的名称。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystem): 文件系统。用于文件的存储和读取。由系统自动传入。
        """
        super().__init__(
            name=name,
            # buffer=buffer, # 直接由系统给出
            hyper_params=hyper_params,
            communicator=communicator,
            data_module=data_module,
            file_system=file_system
        )
        self._communicator.logger_info('TD3BC:Create TD3BC')
        

        self._n_update_times = 0
        
        self._actor = actor() # 在每个算法中，_actor是可以直接用于环境交互的
        self._target_actor = actor()
        
        self._q_critic1 = q_critic()
        self._q_critic2 = q_critic()
        
        self._target_q_critic1 = q_critic()
        self._target_q_critic2 = q_critic()

        self.model_dict = {'actor': self._actor,
                           'q_critic1': self._q_critic1,
                           'q_critic2': self._q_critic2,
                           }
        self._weight_path = weight_path

        # TODO 这俩最好从环境获取
        self._action_min = -1
        self._action_max = 1
        
    def init(self, buffer: BufferBase):
        """
        初始化TD3BC算法。
        """
        self._communicator.logger_info('TD3BC Init TD3BC')

        self._buffer = buffer
        # 在新的架构中我们将学习算法和交互算法分开了，所以这个部分只会在学习算法中执行改部分
        
        # 初始化网络模型
        self.initialize_network(self._actor)
        self.initialize_network(self._target_actor)
        
        self.initialize_network(self._q_critic1)
        self.initialize_network(self._q_critic2)
        
        self.initialize_network(self._target_q_critic1)
        self.initialize_network(self._target_q_critic2)

        # 加载初始权重
        for model_name, model in self.model_dict:
            try:
                model.load_weights(self._weight_path + '/' + model_name + '.h5')
                self._communicator.logger_success('TD3BC: Load initial {} weights'.format(model_name))
            except FileExistsError:
                self._communicator.logger_error('TD3BC: Load initial {} weights error 1'.format(model_name))
            except OSError:
                self._communicator.logger_error('TD3BC: Load initial {} weights error 2'.format(model_name))
        
        # 拷贝网络参数
        self.copy_weights(self._actor, self._target_actor)
        self.copy_weights(self._q_critic1, self._target_q_critic1)
        self.copy_weights(self._q_critic2, self._target_q_critic2)
        
        # 创建actor的优化器
        try:
            optimizer_info = self._actor.optimizer_info
        except AttributeError:
            self._communicator.logger_error('TD3BC: Can not get optimizer info from actor.')
            raise RuntimeError('Can not get optimizer info from actor.')
        
        self.create_optimizer(
            name='actor',
            optimizer_info=optimizer_info
        )
        
        # 创建q_critic的优化器，这里我们只使用一个优化器
        
        try:
            optimizer_info = self._q_critic1.optimizer_info
        except AttributeError:
            self._communicator.logger_error('TD3BC: Can not get optimizer info from q_critic.')
            raise RuntimeError('Can not get optimizer info from q_critic.')
        
        self.create_optimizer(
            name='q_critic',
            optimizer_info=optimizer_info
        )
        
        # 获取动作的最大值和最小值
        # self._action_max, self._action_min = buffer.get_max_min_single('action')
    
    def compute_target_y(
        self,
        next_actor_inputs,
        next_q_critic_inputs,
        rewards,
        masks,
        sigma=0.5,
        noise_clip_range=0.5,
        gamma=0.99,
    ):
        """
        计算target y。

        Args:
            next_actor_inputs: 下一个状态的actor输入。
            next_q_critic_inputs: 下一个状态的q_critic输入。
            rewards: 奖励。
            masks: 是否结束的标志。
            sigma: 噪声的标准差。
            noise_clip_range: 噪声的裁剪范围。
            gamma: 折扣因子。

        Returns:
            target y。
        """
        next_actions = self._target_actor(next_actor_inputs)
        noise = tf.clip_by_value(
            tf.random.normal(tf.shape(next_actions), stddev=sigma),
            -noise_clip_range,
            noise_clip_range,
        )
        next_actions = tf.clip_by_value(next_actions + noise, self._action_min, self._action_max)
        target_q1 = self._target_q_critic1(next_q_critic_inputs, next_actions)
        target_q2 = self._target_q_critic2(next_q_critic_inputs, next_actions)
        target_q = tf.minimum(target_q1, target_q2)
        target_y = rewards + gamma * target_q * masks
        return target_y
    
    def train_critic1(self,
                      current_q_critic_inputs,
                      current_action,
                      target_y):
        
        """
        训练critic1。

        Args:
            current_q_critic_inputs: 当前状态的q_critic输入。
            current_action: 当前动作。
            target_y: target y。
        
        """
        
        with tf.GradientTape() as tape:
            tape.watch(self._q_critic1.trainable_variables)
            
            q1 = self._q_critic1(current_q_critic_inputs, current_action)
            loss = tf.reduce_mean(tf.square(q1 - target_y))
        
        grads = tape.gradient(loss, self._q_critic1.trainable_variables)
        
        self.q_critic_optimizer.apply_gradients(zip(grads, self._q_critic1.trainable_variables))
        
        optimize_info = {
            'q_critic1_loss': loss.numpy()
        }
        
        return optimize_info
    
    
    def train_critic2(self,
                        current_q_critic_inputs,
                        current_action,
                        target_y):
        """
        训练critic2。

        Args:
            current_q_critic_inputs: 当前状态的q_critic输入。
            current_action: 当前动作。
            target_y: target y。
        """
            
        with tf.GradientTape() as tape:
            tape.watch(self._q_critic2.trainable_variables)
            
            q2 = self._q_critic2(current_q_critic_inputs, current_action)
            loss = tf.reduce_mean(tf.square(q2 - target_y))
        
        grads = tape.gradient(loss, self._q_critic2.trainable_variables)
        
        self.q_critic_optimizer.apply_gradients(zip(grads, self._q_critic2.trainable_variables))
        
        optimize_info = {
            'q_critic2_loss': loss.numpy()
        }
        
        return optimize_info
    
    def train_actor(
        self,
        current_actor_inputs,
        target_action,
        alpha=0.2,
    ):
        """
        训练actor。
        
        Args:
            current_actor_inputs: 当前状态的actor输入。
            target_action: target action。
            alpha: 学习率。
        """
        
        with tf.GradientTape() as tape:
            tape.watch(self._actor.trainable_variables)
            
            current_action = self._actor(current_actor_inputs)
            
            q1 = self._q_critic1(current_actor_inputs, current_action)
            
            lamb = alpha / tf.reduce_mean(
                tf.math.abs(
                    self._q_critic1(current_actor_inputs, current_action)
                )
            )
            
            q_loss = tf.reduce_mean(q1)
            bc_loss = tf.reduce_mean(tf.square(current_action - target_action))
            
            loss = -lamb * q_loss + bc_loss
            
            
        grads = tape.gradient(loss, self._actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self._actor.trainable_variables))
        
        optimize_info = {
            'actor_loss': loss.numpy(),
            'actor_q_loss': q_loss.numpy(),
            'actor_bc_loss': bc_loss.numpy(),
            'lamb': lamb.numpy()
        }
        
        return optimize_info
    
    def optimize(self, buffer:BufferBase) -> LossTracker:

        for _ in range(self._hyper_params.update_times):
            sample_data_dict = buffer.sample_data(self._hyper_params.batch_size)
            self._communicator.logger_info('sample_data_dict:{}'.format(sample_data_dict))
            next_actor_inputs = buffer.get_corresponding_data(
                data_dict=sample_data_dict,
                data_names=self._actor.input_names,
                prefix='next_',
                convert_to_tensor=True
            )[0]

            next_q_critic_inputs = buffer.get_corresponding_data(
                data_dict=sample_data_dict,
                data_names=self._q_critic1.input_names,
                prefix='next_',
                convert_to_tensor=True,
                filter=['action']
            )[0]

            current_actor_inputs = buffer.get_corresponding_data(
                data_dict=sample_data_dict,
                data_names=self._actor.input_names,
                convert_to_tensor=True
            )[0]

            current_q_critic_inputs = buffer.get_corresponding_data(
                data_dict=sample_data_dict,
                data_names=self._q_critic1.input_names,
                convert_to_tensor=True,
                filter=['action']
            )[0]

            rewards = tf.convert_to_tensor(sample_data_dict['env_reward'], dtype=tf.float32)
            masks = tf.convert_to_tensor(sample_data_dict['mask'], dtype=tf.float32)
            current_action = tf.convert_to_tensor(sample_data_dict['action'], dtype=tf.float32)

            target_y = self.compute_target_y(
                next_actor_inputs=next_actor_inputs,
                next_q_critic_inputs=next_q_critic_inputs,
                rewards=rewards,
                masks=masks,
                sigma=self._hyper_params.sigma,
                noise_clip_range=self._hyper_params.noise_clip_range,
                gamma=self._hyper_params.gamma
            )
            
            q1_optimize_info = self.train_critic1(
                current_q_critic_inputs=current_q_critic_inputs,
                current_action=current_action,
                target_y=target_y
            )
            self._loss_tracker.add_data(q1_optimize_info)
            
            q2_optimize_info = self.train_critic2(
                current_q_critic_inputs=current_q_critic_inputs,
                current_action=current_action,
                target_y=target_y
            )
            self._loss_tracker.add_data(q2_optimize_info)
            
            self._n_update_times += 1
            
            if self._n_update_times % self._hyper_params.delay_update == 0:
                actor_optimize_info = self.train_actor(
                    current_actor_inputs=current_actor_inputs,
                    target_action=current_action,
                    alpha=self._hyper_params.alpha
                )
                self._loss_tracker.add_data(actor_optimize_info)
                
                self.soft_update_weights(
                    source_model=self._actor,
                    target_model=self._target_actor,
                    tau=self._hyper_params.tau
                )
                
                self.soft_update_weights(
                    source_model=self._q_critic1,
                    target_model=self._target_q_critic1,
                    tau=self._hyper_params.tau
                )
                
                self.soft_update_weights(
                    source_model=self._q_critic2,
                    target_model=self._target_q_critic2,
                    tau=self._hyper_params.tau
                )
                
        return self._loss_tracker
            
            
            
    