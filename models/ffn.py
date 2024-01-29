import torch.nn as nn

from .mlp import MLP, PosEncoding, RandomFourierFeatures


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

        if self.ffn_configs.rff:
            self.encoding = RandomFourierFeatures(dim_in, num_frequencies, sigma=self.ffn_configs.rff_std)
        else:
            self.encoding = PosEncoding(dim_in, num_frequencies)

        self.net = MLP(
            dim_in=self.encoding.out_dim,
            dim_out=dim_out,
            mlp_configs=ffn_configs
        )

        # self.activation = nn.Sigmoid()

    def forward(self, x, label=None):
        x = self.encoding(x)
        x = self.net(x)
        # x = self.activation(x)
        return x
    