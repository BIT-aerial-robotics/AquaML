# AquaML

## 使用教程

### Meta Gradient Reinforcement learning

#### 环境创建创建指南

在使用该算法时，如果不需要reward超参数调节，正常创建即可。如果要是用meta调整reward参数，请参照以下方式。

首先需要声明元学习参数是什么，我们给出的接口是``self.meta_parameters``，注意，字典里面的key应该是env的一个属性例如：

```python
self.ratio = 1.0
self.bias = 0.0

self.meta_parameters = {
          'ratio': 1.0,
           'bias': 0.0,
        }
```

此外需要声明``get_reward``的输入信息：

```python
self.reward_fn_input = ('indicate_reward', 'ratio', 'bias')
```

在meta里面计算reward时候会自动调用``get_reward``函数。注意``get_reward``输入是是tensor中间所有运算都要用tensorflow提供的接口运算。例如:

```python
def get_reward(self, indicate_reward, ratio, bias):
      new_reward = ratio * (reward + bias)
      return new_reward
```

注意：``get_reward``输入必须和``reward_fn_input``。

## Tricks

该框架属于分模块设计可以很轻松的用一些技巧。

### 1. 学习率衰减

该框架能够使用tensorflow提供的所有学习率衰减方法，只需要在定义模型时候将``self.learning_rate``替换为tensorflow的``tf.keras.optimizers.schedules``, 更多信息请参考https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

## 提醒

1. 当前算法属于不稳定阶段，存在一些bug，当运行多线程时候，检查cache下是否有模型文件,如果没有说明有bug。
2. FusionPPO里面不推荐使用batch advantage normalization。

## 更新说明

### v2.0.1

1. 添加meta算法。
2. 添加PPO tricks例如batch advantage normalization。