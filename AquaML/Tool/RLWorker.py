from AquaML.args.RLArgs import EnvArgs
from AquaML import RLPolicyManager
from AquaML import DataManager


class RLWorker:
    def __init__(self, env_args: EnvArgs, policy: RLPolicyManager, dara_manager: DataManager, env):
        self.env_args = env_args
        self.policy = policy
        self.dara_manager = dara_manager
        self.step_count = 0
        self.reset_flag = True
        self.env = env

        self.obs = None

    def step(self):
        """
        Run one step, policy interact with env.

        :return:
        """
        if self.reset_flag:
            self.obs = self.env.reset()
            # self.dara_manager.reset()
            self.reset_flag = False
            self.step_count = 0
            self.policy.reset_actor(1)
        # else:
        #     obs = obs_

        action = self.policy.get_action(self.obs)

        obs_, reward, done = self.env.step(action['action'])

        self.step_count += 1

        if self.step_count >= self.env_args.max_steps:
            clip_mask = 0
            mask = 0
            done = True
        else:
            if done:
                mask = 0
                clip_mask = 0
            else:
                mask = 1
                if self.step_count % self.env_args.episode_clip == 0:
                    clip_mask = 0
                else:
                    clip_mask = 1

        self.dara_manager.store(self.obs, action, reward, obs_, mask, clip_mask)

        if done:
            self.reset_flag = True

        self.obs = obs_

    def reset(self):
        self.step_count = 0

    def roll(self):
        # TODO: extend to multi process.
        for _ in range(self.env_args.one_thread_total_steps):
            self.step()
        self.dara_manager.reset()
