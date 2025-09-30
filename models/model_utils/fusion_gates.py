import torch
from torch import nn


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, feature_list, **kwargs):

        a = feature_list[0]
        v = feature_list[1]

        if self.x_film:
            film = a
            to_be_film = v
        else:
            film = v
            to_be_film = a

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        if "detach_a" in kwargs and kwargs["detach_a"] and self.x_film:
            gamma = gamma.detach()
            beta = beta.detach()
        if "detach_a" in kwargs and kwargs["detach_a"] and not self.x_film:
            to_be_film = to_be_film.detach()
        if "detach_v" in kwargs and kwargs["detach_v"] and self.x_film:
            to_be_film = to_be_film.detach()
        if "detach_v" in kwargs and kwargs["detach_v"] and not self.x_film:
            gamma = gamma.detach()
            beta = beta.detach()

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_mod0_x = nn.Linear(input_dim, dim)
        self.fc_mod1_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list, **kwargs):

        x = feature_list[0]
        y = feature_list[1]

        out_x = self.fc_mod0_x(x)
        out_y = self.fc_mod1_y(y)

        if "detach_a" in kwargs and kwargs["detach_a"]:
            out_x = out_x.detach()
        if "detach_v" in kwargs and kwargs["detach_v"]:
            out_y = out_y.detach()

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return output
