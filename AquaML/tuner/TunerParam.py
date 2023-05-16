
class SearchParam:
    def __init__(self, search_list=None, search_start=None, search_end=True, search_step=None):
        
        self.search_list = search_list
        self.search_start = search_start
        self.search_end = search_end
        self.search_step = search_step
        
        if search_list is not None:
            self.search_list = search_list
        else:
            self.seed_list = list(range(self.search_start, self.search_end, self.search_step))
            

class TunerParam:
    
    def __init__(self):
        
        self.seed_list = None
        
        self.hyper_param_search_dict = {}
    
    def random_seed_search(self, seed_list=None, seed_start=None, seed_end=None, seed_step=None):
        """
        随机种子搜索,两种设置模式，直接设置种子列表，或者设置种子范围和步长

        Args:
            seed_list (_type_, optional): 搜索的种子列表. 
            seed_start (_type_, optional): 种子起始值.
            seed_end (_type_, optional): 种子终止值.
            seed_step (_type_, optional): 间隔步长.
        """
        
        # 模式一：直接设置种子列表
        if seed_list is not None:
            self.seed_list = seed_list
        
        else:
               # 模式二：设置种子范围和步长
        
            # 判断seed_start，seed_end，seed_step是否为None
            error_str = ""
            if seed_start is None:
                error_str += "seed_start is None. "
            if seed_end is None:
                error_str += "seed_end is None. "
            if seed_step is None:
                error_str += "seed_step is None. "
        
            if error_str != "":
                raise ValueError(error_str)
            else:
                # 生成种子列表
                self.seed_list = list(range(seed_start, seed_end, seed_step))
    
    def search_hyperparameter_setting(self, search_param:SearchParam, name:str):
        
        search_list = search_param.search_list
        
        self.hyper_param_search_dict[name] = search_list
     
    
    