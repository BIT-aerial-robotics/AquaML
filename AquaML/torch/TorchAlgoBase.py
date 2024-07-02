from AquaML.algo.AlgoBase import AlgoBase
from AquaML.algo.RLAlgoBase import RLAlgoBase
# from AquaML.algo.ModelBase import ModelBase
from AquaML import logger, data_module,settings
import torch
from torch import nn

from AquaML.param.ParamBase import RLParmBase

class TorchAlgoBase(AlgoBase):
    """
    TorchAlgoBase是基于PyTorch的算法的基类。
    """
    
    def __init__(self,
                 hyper_params,
                 model_dict,
                 ):
        """
        TorchAlgoBase的构造函数。
        
        Args:
            hyper_params (ParamBase): 超参数。
            model_dict (dict): 模型字典。
        """
        # super(TFAlgoBase, self).__init__(hyper_params, model_dict)
        pass
    
    def initialize_network(self, model:nn.Module):
        """
        
        该部分用于初始化网络，提前发现网络的结构中的一些问题，对算法进行检查。

        Args:
            model (nn.Module): 模型。
        """
        
        input_data_names = model.input_names
        
        # create tensor according to input data name
        input_data = []
        
        for name in input_data_names:
            
            # 从data_module中获取数据信息
            data_info = data_module.query_data(
                name=name,
                set_name=self._algo_name
            )[0]
            
            data = torch.zeros(data_info.shape, dtype=torch.float32)
            data = torch.unsqueeze(data, dim=0).to(settings.device)
            
            input_data.append(data)

        with torch.no_grad():
            model(*input_data) # 如果模型有多个输入，需要将输入数据展开
            
    
    def create_optimizer(self, model: nn.Module, other_params=None):
        """
        创建优化器。
        
        Args:
            model (ModelBase): 模型。
            other_params (list, optional): 其他参数。默认为None。
        """
        opt_type = model.optimizer_type
        learning_rate = model.learning_rate
        args = model.optimizer_other_args
        
        clipnorm = None
        
        if 'clipnorm' in args:
            clipnorm = args['clipnorm']
            args.pop('clipnorm')
            # dict().pop
        
        
        if other_params is not None:
            
            opt_param = [{'params': model.parameters()}]       

            if isinstance(other_params, list):
                for param in other_params:
                    opt_param.append({'params': param})
            else:
                opt_param.append({'params': other_params})
        else:
            opt_param = model.parameters()
        
        optimizer = getattr(torch.optim, opt_type)(
            params = opt_param,
            lr=learning_rate,
            **args
        )
        
        if clipnorm is not None:
            def optimizer_step(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                optimizer.step()
                optimizer.zero_grad()
        else:
            def optimizer_step(loss):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        return optimizer, optimizer_step
    
class TorchRLAlgoBase(TorchAlgoBase, RLAlgoBase):
    """
    TorchRLAlgoBase是基于PyTorch的强化学习算法的基类。
    """
    
    def __init__(self, hyper_params: RLParmBase, model_dict):
        """
        TorchRLAlgoBase的构造函数。
        
        Args:
            hyper_params (RLParmBase): 超参数。
            model_dict (dict): 模型字典。
        """
        TorchAlgoBase.__init__(self, hyper_params, model_dict)
        RLAlgoBase.__init__(self, hyper_params, model_dict)
        
    def init(self):
        
        self._action_size = settings.env_num

        
        for name, model in self._model_dict.items():
            self.initialize_network(model)