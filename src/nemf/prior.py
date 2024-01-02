import torch
import torch.nn as nn

from .residual_blocks import ResidualBlock, SkeletonResidual, residual_ratio
from .skeleton import SkeletonConv, SkeletonPool, find_neighbor


class LocalEncoder(nn.Module):
    def __init__(self, args, topology):
        super(LocalEncoder, self).__init__()
        self.topologies = [topology]
        self.channel_base = [args.channel_base]

        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        # self.convs = []

        kernel_size = args.kernel_size
        kernel_even = False if kernel_size % 2 else True
        padding = (kernel_size - 1) // 2
        bias = True

        for _ in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(args.num_layers):
            seq = []
            neighbour_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            if i == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            last_pool = True if i == args.num_layers - 1 else False

            # (T, J, D) => (T, J', D)
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbour_list), last_pool=last_pool)

            if args.use_residual_blocks:
                # (T, J, D) => (T/2, J', 2D)
                seq.append(SkeletonResidual(self.topologies[i], neighbour_list, joint_num=self.edge_num[i], in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=2, padding=padding, padding_mode=args.padding_mode, bias=bias,
                                            extra_conv=args.extra_conv, pooling_mode=args.skeleton_pool, activation=args.activation, last_pool=last_pool))
            else:
                for _ in range(args.extra_conv):
                    # (T, J, D) => (T, J, D)
                    seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=in_channels,
                                            joint_num=self.edge_num[i], kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                                            stride=1,
                                            padding=padding, padding_mode=args.padding_mode, bias=bias))
                    seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
                # (T, J, D) => (T/2, J, 2D)
                seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=out_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=False,
                                        in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
                # self.convs.append(seq[-1])

                seq.append(pool)
                seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

        in_features = self.channel_base[-1] * len(self.pooling_list[-1])
        in_features *= args.temporal_scale
        self.mu = nn.Linear(in_features, args.z_dim)
        self.logvar = nn.Linear(in_features, args.z_dim)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = output.view(output.shape[0], -1)

        return self.mu(output), self.logvar(output)


class GlobalEncoder(nn.Module):
    def __init__(self, args):
        super(GlobalEncoder, self).__init__()

        in_channels = args.in_channels  # root orientation + root velocity
        out_channels = 512
        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        self.encoder = nn.Sequential(
            ResidualBlock(in_channels=in_channels, out_channels=128, kernel_size=kernel_size, stride=2, padding=padding, residual_ratio=residual_ratio(1), activation=args.activation),
            ResidualBlock(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=2, padding=padding, residual_ratio=residual_ratio(2), activation=args.activation),
            ResidualBlock(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=2, padding=padding, residual_ratio=residual_ratio(3), activation=args.activation),
            ResidualBlock(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=padding, residual_ratio=residual_ratio(4), activation=args.activation)
        )

        in_features = out_channels * args.temporal_scale
        self.mu = nn.Linear(in_features, args.z_dim)
        self.logvar = nn.Linear(in_features, args.z_dim)

    def forward(self, input):
        feature = self.encoder(input)
        feature = feature.view(feature.shape[0], -1)

        return self.mu(feature), self.logvar(feature)
