import raisimpy as raisim
import numpy as np
import os
import time
from gym.spaces import Box
import pdb
import math

"""
gc =  [ base_pos_x, base_pos_y, base_pos_z , 1., 0., 0., 0., 
       1., 0., 0., 0.,
       extension_uav2_pos,
       1., 0., 0., 0.,
       prop_p,
       prop_p,
       prop_p,
       prop_p,
       
       extension_left_pos, 
       extension_uav3_pos,
       1., 0., 0., 0.,
       
       prop_p,
       prop_p,
       prop_p,
       prop_p,
       extension_right_pos,
       prop_p,
       prop_p,
       prop_p,
       prop_p]
                    0           6            3
observation = [base_pos_x, base_pos_y, base_pos_z ,
                extension_uav2_pos,  -1.55
                extension_left_pos,   1.55
                extension_uav3_pos,   -1.6
                extension_right_pos   1.6 
                ]

action = [base_vel_x, base_vel_y, base_vel_z ,
                extension_uav2_vel, 
                extension_left_vel,
                extension_uav3_vel,
                extension_right_vel
                ]
"""

"""
    
    action_space : [-0.1, 0.1]
    collision: done
    
"""


class Amorphosis_env():
    def __init__(self,
                 file="/home/fyt/project/test_raisim/rsc/huan_line_0825/huan_line.urdf",  # urdf所在路径
                 max_step=600,
                 dt=0.1
                 ):
        self.dt = dt
        self.max_step = max_step
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../activation.raisim")
        self.world = raisim.World()
        self.world.setTimeStep(self.dt)  # 设置模拟器的固定时间步长
        self.world.addGround()

        self.server = raisim.RaisimServer(self.world)
        self.server.launchServer(8080)
        self.agent = self.world.addArticulatedSystem(file)
        self.agent.setName("huan")
        self.server.focusOn(self.agent)
        self.bodyidx = self.agent.getBodyIdx("base_link")
        self.prop_p = 0.
        self.collision = False
        self.rew_coeff = {"pos": 1.,
                          "formation": 0.01,
                          "collision": 2500,
                          "action_cost": 0.001}
        self.add_box()
        self.reset()

        self._max_episode_steps = max_step

    def add_box(self, ):
        box1 = self.world.addBox(10, 2.6, 50, 100, "steel", 63)
        # box1.setPosition(-6.3, 4.3, 0)
        box1.setPosition(-5.8, 4.3, 0)
        box1.setBodyType(raisim.BodyType.STATIC)

        box2 = self.world.addBox(10, 2.6, 50, 100, "steel", 63)
        box2.setPosition(5.4, 4.3, 0)
        box2.setBodyType(raisim.BodyType.STATIC)

        box3 = self.world.addBox(0.3, 1.6, 50, 100, "steel", 63)
        box3.setPosition(10., 4., 0)
        box3.setBodyType(raisim.BodyType.STATIC)

    # def vertualBox(self,):
    #     box1 = 

    def rotate_coordinates(self, x0, y0, theta_deg, r=0.528):
        theta_rad = math.radians(theta_deg)
        x = r * math.cos(theta_rad) + x0
        y = r * math.sin(theta_rad) + y0
        return x, y

    def judgeVertualCollision(self, ):
        left_rod_lpos = self.agent.getPosition(8)
        ratation_deg = self.extension_left_pos
        left_rod_rpos = self.rotate_coordinates(left_rod_lpos[0], left_rod_lpos[1], ratation_deg)

        right_rod_rpos = self.agent.getPosition(15)
        ratation_deg = self.extension_right_pos
        left_rod_rpos = self.rotate_coordinates(right_rod_rpos[0], right_rod_rpos[1], ratation_deg)

    def step(self, action=np.zeros(7)):
        self.action = action
        self.base_pos_x += 0. * self.dt
        self.base_pos_y += 0.1 * self.dt
        self.base_pos_z += 0. * self.dt

        self.extension_uav2_pos = -1.22
        self.extension_left_pos = 1.22
        self.extension_uav3_pos = -1.22
        self.extension_right_pos = 1.22

        self.extension_uav2_pos += action[3] * self.dt
        self.extension_left_pos += action[4] * self.dt
        self.extension_uav3_pos += action[5] * self.dt
        self.extension_right_pos += action[6] * self.dt

        self.prop_p += self.prop_vel * self.step_count
        self.observation = np.array([self.base_pos_x, self.base_pos_y, self.base_pos_z,
                                     self.extension_uav2_pos,
                                     self.extension_left_pos,
                                     self.extension_uav3_pos,
                                     self.extension_right_pos])

        gc = [self.base_pos_x, self.base_pos_y, self.base_pos_z, 1., 0., 0., 0.,
              1., 0., 0., 0.,
              self.extension_uav2_pos,
              1., 0., 0., 0.,
              self.prop_p,
              self.prop_p,
              self.prop_p,
              self.prop_p,

              self.extension_left_pos,
              self.extension_uav3_pos,
              1., 0., 0., 0.,

              self.prop_p,
              self.prop_p,
              self.prop_p,
              self.prop_p,
              self.extension_right_pos,
              self.prop_p,
              self.prop_p,
              self.prop_p,
              self.prop_p]

        self.agent.setGeneralizedVelocity(np.zeros(31))
        self.agent.setGeneralizedCoordinate(gc)
        self.world.integrate()
        self.step_count += 1
        reward, info = self.getReward()
        done = False
        if self.step_count >= self._max_episode_steps or self.collision:
            done = True
        return self.observation, reward, done, info

    """
    这里如果collision最好把整个训练环境done掉,开始新的epoch,否则可能会影响采样速度,raisim也可能会挂掉
    """

    def getReward(self, ):
        gc_now = self.observation
        reward_pos = self.rew_coeff["pos"] * np.linalg.norm(gc_now[:3] - self.goal[:3])
        reward_formation = self.rew_coeff["formation"] * np.linalg.norm(gc_now[3:] - self.goal[3:])
        reward_action = self.rew_coeff["action_cost"] * np.linalg.norm(self.action)
        self.collision = self.judgeCollision()
        reward_collision = self.rew_coeff["collision"] * int(self.collision)
        reward = -np.sum([reward_pos, reward_formation, reward_collision, reward_action])

        info = {"reward_pos": reward_pos,
                "reward_formation": reward_formation,
                "reward_collision": reward_collision,
                "action_cost": reward_action}
        return reward, info

    def get_state(self, ):
        gc = self.agent.getGeneralizedCoordinate()
        self.base_pos_x = gc[0]
        self.base_pos_y = gc[1]
        self.base_pos_z = gc[2]
        self.extension_uav2_pos = gc[11]
        self.extension_left_pos = gc[20]
        self.extension_uav3_pos = gc[21]
        self.extension_right_pos = gc[30]

    def judgeCollision(self, ):
        forces = []
        torques = []
        for contact in self.agent.getContacts():
            # pdb.set_trace()
            print("have collision")
            return True
        return False

    def reset(self, ):
        self.step_count = 0.
        # self.goal = np.array([0., 8., 3.,
        #                         -1.55,
        #                         1.55,
        #                         -1.6,
        #                         1.6])
        self.goal = np.array([0., 6., 3.,
                              0,
                              0,
                              0,
                              0])
        # 初始位置在空中，z为3.
        gc = [0., 0., 3., 1., 0., 0., 0.,
              1., 0., 0., 0.,
              0.,
              1., 0., 0., 0.,
              0.,
              0.,
              0.,
              0.,

              0.,
              0.,
              1., 0., 0., 0.,

              0.,
              0.,
              0.,
              0.,
              0.,
              0.,
              0.,
              0.,
              0.]

        self.agent.setGeneralizedVelocity(np.zeros(31))
        self.agent.setGeneralizedCoordinate(gc)
        self.collision = False
        self.prop_vel = 0.001
        self.prop_p = 0.
        self.base_pos_x = 0.
        self.base_pos_y = 0.
        self.base_pos_z = 3.
        self.extension_uav2_pos = 0.
        self.extension_left_pos = 0.
        self.extension_uav3_pos = 0.
        self.extension_right_pos = 0.
        self.observation = [0., 0., 3., 0., 0., 0., 0.]
        self.world.integrate()
        return self.observation

    @property
    def action_space(self):
        high = 1. * np.ones(7)
        low = -high
        return Box(low, high)

    @property
    def observation_space(self):
        high = np.infty * np.ones(7)
        low = -high
        return Box(low, high)

    def close(self, ):
        self.server.killServer()

    def get_reward(self, is_collision, angle, collision_reward=True):

        # 启用避障reward
        if collision_reward:
            if is_collision:
                return -1
            else:
                return 0
        else:
            # 启用环形reward
            reward = -np.linalg.norm(angle)
            return reward

    # 双足机器人避障reward
    def complicate_reward1(self, is_collision, angle, collision_reward=True):
        """
        碰撞到了就done

        """

        if is_collision:
            return -1
        else:
            return 0.05


if __name__ == '__main__':

    env = Amorphosis_env()
    pdb.set_trace()
    for _ in range(10000):
        state, done = env.reset(), False
        rewards = 0
        step = 0
        i = 0
        next_state = state
        while not done:
            i += 1
            print(i)
            action = env.action_space.sample()
            action[3:] = np.zeros(4)
            time.sleep(0.01)
            # action[5] = 0.

            # if next_state[-1] > 3.14 or next_state[-1] < -3.14:
            #     action[-1] = 0
            print(f"action is {action}")
            print(f"state is {next_state}")
            print(env.agent.getGeneralizedCoordinate())
            next_state, reward, done, info = env.step(action)

            rewards += reward
            step += 1
        print(step)
    env.close()
    print("finish")
