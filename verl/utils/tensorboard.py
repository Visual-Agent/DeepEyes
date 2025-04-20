import os
from torch.utils.tensorboard import SummaryWriter
from verl import DataProto


class TensorboardLogger:
    
    def __init__(
        self,
        root_log_dir: str,
        project_name: str,
        experiment_name: str
    ):
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                root_log_dir, 
                project_name, 
                experiment_name
            )
        )

    def log(
        self,
        data: dict,
        step: int,
        *args,
        **kwargs
    ):
        for k, v in data.items():
            try:
                self.writer.add_scalar(k, v, step)
            except:
                print("[TensorboardLogger] Failed to log key:", k, ", skipped.")