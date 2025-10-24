import torch.distributed as dist


class DistributedHelper:
    #Helper class for distributed training operations.
    
    @staticmethod
    def get_world_size():
        #Get world size safely.
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1
    
    @staticmethod
    def get_rank():
        #Get current rank safely.
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0
    
    @staticmethod
    def is_main_process() -> bool:
        #Check if current process is main.
        return DistributedHelper.get_rank() == 0

