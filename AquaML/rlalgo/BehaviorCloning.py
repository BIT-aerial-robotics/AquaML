from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.rlalgo.Parameters import BehaviorCloning_parameter
from AquaML.DataType import RLIOInfo
from AquaML.rlalgo.ExplorePolicy import VoidExplorePolicy
import tensorflow as tf
import warnings
import copy


class BehaviorCloning(BaseRLAlgo):
    def __init__(self,
                 env,  # 用于采样或者评估策略的环境
                 rl_io_info: RLIOInfo,  # 环境的输入输出信息
                 parameters: BehaviorCloning_parameter,  # 算法参数
                 learner,  # 算法的学习器,一个类
                 actor,  # 算法的专家,一个类
                 computer_type: str = 'PC',  # 计算类型
                 prefix_name: str = None,  # 算法前缀名
                 name: str = 'BehaviorCloning',  # 算法名
                 level: int = 0,  # 算法层级
                 thread_id: int = 0,  # 线程id
                 total_threads: int = 1,  # 总线程数
                 ):
        super().__init__(
            env=env,
            rl_io_info=rl_io_info,
            name=name,
            hyper_parameters=parameters,
            computer_type=computer_type,
            level=level,
            thread_ID=thread_id,
            total_threads=total_threads,
            prefix_name=prefix_name,
        )

        # 当前BC仅支持export policy为lstm的情况
        # expert可以包含lstm，expert当成全局actor
        self.actor = actor()
        self.initialize_actor_config()
        self.initialize_model_weights(self.actor, self.rnn_actor_flag)

        # learner不能包含lstm
        self.learner = learner()
        self.initialize_model_weights(self.learner)

        self._all_model_dict = {
            'learner': self.learner,
            'actor':self.actor
        }

        self._sync_model_dict['learner'] = self.learner

        # create optimizer
        if self.level == 0:
            self.create_optimizer(name='learner', optimizer=self.learner.optimizer, lr=self.learner.learning_rate)
        else:
            self.learner_optimizer = None

        self.explore_policy = VoidExplorePolicy(shape=self.rl_io_info.actor_out_info['action'])

    @tf.function
    def train_learner(self, learner_obs: tuple, target_action: tf.Tensor):

        with tf.GradientTape() as tape:
            out = self.learner(*learner_obs)

            action = out[0]

            loss = tf.reduce_mean(tf.square(action - target_action))

        grads = tape.gradient(loss, self.learner.trainable_variables)
        self.learner_optimizer.apply_gradients(zip(grads, self.learner.trainable_variables))

        dic = {
            'loss': loss,
        }

        return dic

    def _optimize_(self):

        data_dict = self.get_all_data

        learner_obs = self.get_corresponding_data(
            data_dict=data_dict,
            names=self.learner.input_name,
        )

        target_action = data_dict['action']

        target_action = tf.convert_to_tensor(target_action, dtype=tf.float32)

        train_learner_dict = {
            'learner_obs': learner_obs,
            'target_action': target_action,
        }

        batch_size = self.hyper_parameters.batch_size
        buffer_size = self.hyper_parameters.buffer_size

        learner_optimize_info = []

        for _ in range(self.hyper_parameters.update_times):
            start_index = 0
            end_index = 0

            while end_index < buffer_size:
                end_index = min(start_index + batch_size,
                                buffer_size)

                batch_train_learner_input = self.get_batch_data(train_learner_dict, start_index, end_index)

                start_index = end_index

                train_info = self.train_learner(
                    learner_obs=batch_train_learner_input['learner_obs'],
                    target_action=batch_train_learner_input['target_action'],
                )

                learner_optimize_info.append(train_info)

        info = self.cal_average_batch_dict(learner_optimize_info)

        return info






