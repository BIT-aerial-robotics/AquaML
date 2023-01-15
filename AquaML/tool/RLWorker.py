from AquaML.data.DataPool import DataPool
from AquaML.DataType import RLIOInfo 

#TODO: remove in the future
#TODO: 太过于复杂，需要重构
from AquaML.rlalgo.BaseRLAlgo import BaseRLalgo 

class RLWorker:
    
    """ RLWorker is used to interact with environment and get data.
    
    It can be used in on-policy and off-policy reinforcement learning.
    
    """

    def __init__(self, rl_algo:BaseRLalgo):
        self.rl_algo = rl_algo
        self.reset_flag = True
        
        # get information from rl_algo
        self.env = rl_algo.env
        
        
        self.obs = None
        self.step_count = 0
        
    def step(self):
        
        # reset the environment and actor model
        # when first step or reset flag is True
        if self.reset_flag:
            self.obs = self.env.reset()
            self.reset_flag = False
            self.step_count = 0
            self.rl_algo.actor.reset()
        
        action_dict = self.rl_algo.get_action_train(self.obs)
        
        obs_, reward, done, info = self.env.step(action_dict['action']) # obs, reward are dict
        
        self.step_count += 1
        
        # done flag is True, need to reset the environment
        if done:
            self.reset_flag = True
            mask = 0
        else:
            mask = 1
            
        # store the data
        self.rl_algo.store_data(obs=self.obs,action=action_dict,
                                reward=reward, next_obs=obs_, mask=mask)
        
    
    def roll(self):
        """roll the environment and get data.
        when step_count == update_interval, need to update the model
        """
        
        for _ in range(self.rl_algo.each_thread_update_interval):
            self.step()
        