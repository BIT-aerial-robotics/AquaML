import torch

dtype = torch.float

str_dtype = str(dtype)

eval_dtype = eval(str_dtype)
print(eval_dtype)
