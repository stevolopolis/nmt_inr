import torch.nn as nn

from .mlp import MLP, PosEncoding


class FFN(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ffn_configs
    ):
        super().__init__()
        self.ffn_configs = ffn_configs.NET
        self.dim_in = dim_in
        self.dim_out = dim_out
        num_frequencies = self.ffn_configs.num_frequencies
        fourier_features = 2*dim_in*num_frequencies+dim_in

        self.net = MLP(
            dim_in=fourier_features,
            dim_out=dim_out,
            mlp_configs=ffn_configs
        )
        self.encoding = PosEncoding(dim_in, num_frequencies)

    def forward(self, x, label=None):
        x = self.encoding(x)
        x = self.net(x)
        return x
    

