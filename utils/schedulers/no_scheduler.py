import wandb


class No_Scheduler:
    def __init__(self, base_lr):
        self.lr_history = []
        self.base_lr = base_lr
        print("No scheduler is used in this training")

    def step(self, step=None, loss=None):
        self.lr_history.append(self.base_lr)
        wandb.log({"lr": self.base_lr})

    def state_dict(self):
        return {"base_lr": self.base_lr}

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict["base_lr"]
        # self.__dict__.update(state_dict)
