import tensorflow as tf
from AquaML.algo_base.OfflineRLBase import OfflineRLBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import FileSystemBase
from AquaML.core.Tool import LossTracker
from AquaML.param.OfflineRL import IQLParams
from AquaML.buffer.BufferBase import BufferBase


class IQL(OfflineRLBase):
    """
    IQL算法。

    """

    def __init__(self,
                 actor: tf.keras.Model,
                 q_critic: tf.keras.Model,
                 state_value: tf.keras.Model,
                 hyper_params: IQLParams,
                 weight_path: str,
                 # buffer,
                 communicator: CommunicatorBase,
                 data_module: DataModule,
                 file_system: FileSystemBase,
                 name: str = 'IQL'
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
        self._communicator.logger_info('IQL:Create IQL')

        self._n_update_times = 0

        self._actor = actor()  # 在每个算法中，_actor是可以直接用于环境交互的
        self._target_actor = actor()

        self._q_critic = q_critic()
        self._target_q_critic = q_critic()

        self._state_value = state_value()

        self.model_dict = {'actor': self._actor,
                           'q_critic': self._q_critic,
                           's_value': self._state_value,
                           }
        self._weight_path = weight_path

        # TODO 这俩最好从环境获取
        self._action_min = -1
        self._action_max = 1

    def init(self, buffer: BufferBase):
        """
        初始化IQL算法。
        """
        self._communicator.logger_info('IQL: Init IQL')

        self._buffer = buffer
        # 在新的架构中我们将学习算法和交互算法分开了，所以这个部分只会在学习算法中执行改部分

        # 初始化网络模型
        self.initialize_network(self._actor)
        self.initialize_network(self._target_actor)

        self.initialize_network(self._q_critic)
        self.initialize_network(self._target_q_critic)

        self.initialize_network(self._state_value)

        # 加载初始权重
        for model_name, model in self.model_dict.items():
            try:
                model.load_weights(self._weight_path + '/' + model_name + '.h5')
                self._communicator.logger_success('IQL: Load initial {} weights'.format(model_name))
            except FileExistsError:
                self._communicator.logger_error('IQL: Load initial {} weights error 1'.format(model_name))
            except OSError:
                self._communicator.logger_error('IQL: Load initial {} weights error 2'.format(model_name))


        # 拷贝网络参数
        self.copy_weights(self._actor, self._target_actor)
        self.copy_weights(self._q_critic, self._target_q_critic)

        # 创建actor的优化器
        try:
            optimizer_info = self._actor.optimizer_info
        except AttributeError:
            self._communicator.logger_error('IQL: Can not get optimizer info from actor.')
            raise RuntimeError('Can not get optimizer info from actor.')

        self.create_optimizer(
            name='actor',
            optimizer_info=optimizer_info
        )

        # 创建q_critic的优化器
        try:
            optimizer_info = self._q_critic.optimizer_info
        except AttributeError:
            self._communicator.logger_error('IQL: Can not get optimizer info from q_critic.')
            raise RuntimeError('Can not get optimizer info from q_critic.')

        self.create_optimizer(
            name='q_critic',
            optimizer_info=optimizer_info
        )

        # 创建state_value的优化器
        try:
            optimizer_info = self._state_value.optimizer_info
        except AttributeError:
            self._communicator.logger_error('IQL: Can not get optimizer info from state_value.')
            raise RuntimeError('Can not get optimizer info from state_value.')

        self.create_optimizer(
            name='state_value',
            optimizer_info=optimizer_info
        )

        # # 获取动作的最大值和最小值
        # self._action_max, self._action_min = buffer.get_max_min_single('action')
    @tf.function
    def train_critic(self,
                     current_q_critic_inputs,
                     current_action,
                     rewards,
                     next_q_critic_inputs,
                     masks,
                     gamma=0.99):

        """
        训练critic。

        Args:
            current_q_critic_inputs: 当前状态的q_critic输入。
            current_action: 当前动作。

        """

        next_state_value = self._state_value(next_q_critic_inputs)
        target_y = rewards + gamma * masks * next_state_value
        with tf.GradientTape() as tape:
            tape.watch(self._q_critic.trainable_variables)

            q1, q2 = self._q_critic(current_q_critic_inputs, current_action)
            loss = tf.reduce_mean(tf.square(q1 - target_y) + tf.square(q2 - target_y))

        grads = tape.gradient(loss, self._q_critic.trainable_variables)

        self.q_critic_optimizer.apply_gradients(zip(grads, self._q_critic.trainable_variables))

        optimize_info = {
            'q_critic_loss': loss
        }

        return optimize_info
    @tf.function
    def train_state_value(self,
                          current_actor_inputs,
                          current_action,
                          expectile=0.7):
        """
        训练state_value.

        Args:
            current_actor_inputs: 当前状态的q_critic输入。
            current_action: 当前动作。
            expectile: 非对称损失函数期望。
        """

        with tf.GradientTape() as tape:
            tape.watch(self._state_value.trainable_variables)

            q1, q2 = self._target_q_critic(current_actor_inputs, current_action)
            q = tf.minimum(q1, q2)
            v = self._state_value(current_actor_inputs)
            diff = q - v

            weight = tf.where(diff > 0, expectile, (1 - expectile))
            loss = tf.reduce_mean(weight * tf.square(diff))

        grads = tape.gradient(loss, self._state_value.trainable_variables)

        self.state_value_optimizer.apply_gradients(zip(grads, self._state_value.trainable_variables))

        optimize_info = {
            'state_value_loss': loss
        }

        return optimize_info
    @tf.function
    def train_actor(
            self,
            current_actor_inputs,
            target_action,
            temperature=3,
    ):
        """
        训练actor。

        Args:
            current_actor_inputs: 当前状态的actor输入。
            target_action: target action。
        """

        with tf.GradientTape() as tape:
            tape.watch(self._actor.trainable_variables)

            v = self._state_value(current_actor_inputs)
            q1, q2 = self._target_q_critic(current_actor_inputs, target_action)
            q = tf.minimum(q1, q2)
            exp_a = tf.exp((q - v) * temperature)
            exp_a = tf.clip_by_value(exp_a, -100, 100)

            mu = self._actor(current_actor_inputs)
            loss = tf.reduce_mean(exp_a * tf.square(mu - target_action))

        grads = tape.gradient(loss, self._actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self._actor.trainable_variables))

        optimize_info = {
            'actor_loss': loss
        }

        return optimize_info

    def optimize(self, buffer: BufferBase) -> LossTracker:

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
                data_names=self._q_critic.input_names,
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
                data_names=self._q_critic.input_names,
                convert_to_tensor=True,
                filter=['action']
            )[0]

            rewards = tf.convert_to_tensor(sample_data_dict['env_reward'], dtype=tf.float32)
            masks = tf.convert_to_tensor(sample_data_dict['mask'], dtype=tf.float32)
            current_action = tf.convert_to_tensor(sample_data_dict['action'], dtype=tf.float32)

            v_optimize_info = self.train_state_value(
                current_actor_inputs=current_actor_inputs,
                current_action=current_action,
                expectile=self._hyper_params.expectile
            )
            # self._communicator.logger_success(v_optimize_info)
            loss_dict = self._loss_tracker.add_data(v_optimize_info)
            # self._communicator.logger_success(loss_dict)

            q_optimize_info = self.train_critic(
                current_q_critic_inputs=current_q_critic_inputs,
                current_action=current_action,
                rewards=rewards,
                next_q_critic_inputs=next_q_critic_inputs,
                masks=masks,
                gamma=self._hyper_params.gamma
            )
            self._loss_tracker.add_data(q_optimize_info)

            self._n_update_times += 1

            if self._n_update_times % self._hyper_params.delay_update == 0:
                actor_optimize_info = self.train_actor(
                    current_actor_inputs=current_actor_inputs,
                    target_action=current_action,
                    temperature=self._hyper_params.temperature
                )
                self._loss_tracker.add_data(actor_optimize_info)

                self.soft_update_weights(
                    source_model=self._actor,
                    target_model=self._target_actor,
                    tau=self._hyper_params.tau
                )

                self.soft_update_weights(
                    source_model=self._q_critic,
                    target_model=self._target_q_critic,
                    tau=self._hyper_params.tau
                )

        return self._loss_tracker
