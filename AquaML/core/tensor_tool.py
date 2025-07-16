import torch
from typing import Dict


class TensorTool:
    """
    TensorTool is a tool for processing tensors.
    """

    def __init__(self):
        pass

    def concat_dict_tensors(self, dict_tensors: Dict[str, torch.Tensor], dim: int = 1) -> Dict[str, torch.Tensor]:
        """
        Concatenate a dictionary of tensors along a given dimension.
        """


        states = torch.cat(list(dict_tensors.values()), dim=dim)    

        states_dict = {"states": states}

        return states_dict
    