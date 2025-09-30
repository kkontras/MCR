import torch


class GSPlugin:
    def __init__(self, gs_flag=True):

        super().__init__()

        dtype = torch.cuda.FloatTensor  # run on GPU
        with torch.no_grad():
            # self.Pl = torch.autograd.Variable(torch.eye(768).type(dtype))
            self.Pl = torch.autograd.Variable(torch.eye(768).type(dtype)).requires_grad_(False)
        self.exp_count = 0

    # @torch.no_grad()
    def before_update(self, model, before_batch_input, batch_index, len_dataloader, train_exp_counter):

        lamda = batch_index / len_dataloader + 1
        alpha = 1.0 * 0.1**lamda
        # x_mean = torch.mean(strategy.mb_x, 0, True)
        if train_exp_counter != 0:
            for n, w in model.named_parameters():
                if n == "weight":

                    r = torch.mean(before_batch_input.detach(), 0, True)
                    k = torch.mm(self.Pl, torch.t(r))
                    self.Pl = torch.sub(self.Pl, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))

                    pnorm2 = torch.norm(self.Pl.data, p="fro")

                    self.Pl.data = self.Pl.data / pnorm2
                    w.grad.data = torch.mm(w.grad.data, torch.t(self.Pl.data))
