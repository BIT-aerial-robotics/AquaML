from AquaML import DataManager
from AquaML.policy.ReplayPolicy import ReplayPolicy


class ReplayWorker:
    def __init__(self, policy: ReplayPolicy, data_manager: DataManager, env):
        self.policy = policy
        self.data_manager = data_manager
        self.step_count = 0
        self.env = env

        start_pointer = self.policy.start_pointer
        end_pointer = self.policy.end_pointer

        self.total_step = end_pointer - start_pointer + 1

    def step(self):
        action = self.policy.get_action()

        obs, reward_info, done = self.env.step(action)

        self.data_manager.store(obs, action, reward_info, obs, 0, 0)

    def roll(self):
        # TODO: extend to multi process.
        for _ in range(self.total_step):
            self.step()
        self.data_manager.reset()
