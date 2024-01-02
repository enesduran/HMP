import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, n_functions):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('frequencies', 2.0 ** torch.arange(n_functions))

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: a temporal embedding of `x` of shape [..., n_functions * dim * 2]
        """
        freq = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)

        embedding = torch.zeros(*freq.shape[:-1], freq.shape[-1] * 2).cuda()
        embedding[..., 0::2] = freq.sin()
        embedding[..., 1::2] = freq.cos()

        return embedding


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super(Sine, self).__init__()

        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenBlock(nn.Module):
    def __init__(self, in_features, out_features, w0=30, c=6, is_first=False, use_bias=True, activation=None):
        super(SirenBlock, self).__init__()

        self.in_features = in_features
        self.is_first = is_first

        weight = torch.zeros(out_features, in_features)
        bias = torch.zeros(out_features) if use_bias else None
        self.init(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init(self, weight, bias, c, w0):
        n = self.in_features

        w_std = (1 / n) if self.is_first else (np.sqrt(c / n) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)

        return self.activation(out)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, norm_layer=False, activation=None):
        super(FCBlock, self).__init__()

        self.fc = nn.Linear(in_features, out_features)
        self.residual = (in_features == out_features)  # when the input and output have the same dimensions, build a residual block
        self.norm_layer = nn.LayerNorm(out_features) if norm_layer else None
        self.activation = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x):
        """
        Args:
            x: (B, T, D), features are in the last dimension.
        """
        out = self.fc(x)

        if self.norm_layer is not None:
            out = self.norm_layer(out)

        if self.residual:
            return self.activation(out) + x

        return self.activation(out)


class NeuralMotionField(nn.Module):
    def __init__(self, args):
        super(NeuralMotionField, self).__init__()

        self.args = args

        if args.bandwidth != 0:
            self.positional_encoding = PositionalEncoding(args.bandwidth)
            embedding_dim = args.bandwidth * 2
        else:
            embedding_dim = 1

        hidden_neuron = args.hidden_neuron
        in_features = 1 + args.local_z if args.siren else embedding_dim + args.local_z
        local_in_features = hidden_neuron + in_features if args.skip_connection else hidden_neuron
        global_in_features = local_in_features + args.global_z if args.skip_connection else hidden_neuron

        layers = [
            SirenBlock(in_features, hidden_neuron, is_first=True) if args.siren else FCBlock(in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(local_in_features, hidden_neuron) if args.siren else FCBlock(local_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(local_in_features, hidden_neuron) if args.siren else FCBlock(local_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(local_in_features, hidden_neuron) if args.siren else FCBlock(local_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(local_in_features, hidden_neuron) if args.siren else FCBlock(local_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(local_in_features, hidden_neuron) if args.siren else FCBlock(local_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(local_in_features, hidden_neuron) if args.siren else FCBlock(local_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(local_in_features, hidden_neuron) if args.siren else FCBlock(local_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(global_in_features, hidden_neuron) if args.siren else FCBlock(global_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(global_in_features, hidden_neuron) if args.siren else FCBlock(global_in_features, hidden_neuron, norm_layer=args.norm_layer),
            SirenBlock(global_in_features, hidden_neuron) if args.siren else FCBlock(global_in_features, hidden_neuron, norm_layer=args.norm_layer)
        ]
        self.mlp = nn.ModuleList(layers)
        self.skip_layers = [] if not args.skip_connection else list(range(1, len(self.mlp)))
        self.local_layers = list(range(8))
        self.local_linear = nn.Sequential(nn.Linear(hidden_neuron, args.local_output))
        self.global_layers = list(range(8, len(self.mlp)))
        self.global_linear = nn.Sequential(nn.Linear(hidden_neuron, args.global_output))

    def forward(self, t, z_l=None, z_g=None):
        if self.args.bandwidth != 0 and self.args.siren is False:
            t = self.positional_encoding(t)

        if z_l is not None:
            z_l = z_l.unsqueeze(1).expand(-1, t.shape[1], -1)
            x = torch.cat([t, z_l], dim=-1)
        else:
            x = t

        skip_in = x
        for i in self.local_layers:
            if i in self.skip_layers:
                x = torch.cat((x, skip_in), dim=-1)
            x = self.mlp[i](x)
        local_output = self.local_linear(x)

        if z_g is not None:
            z_g = z_g.unsqueeze(1).expand(-1, t.shape[1], -1)
            for i in self.global_layers:
                if i in self.skip_layers:
                    x = torch.cat((x, skip_in, z_g), dim=-1)
                x = self.mlp[i](x)
            global_output = self.global_linear(x)
        else:
            global_output = None

        return local_output, global_output
