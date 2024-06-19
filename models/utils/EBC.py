import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base



class EBC(nn.Module, base.StepModule):
    def __init__(self, channels, snn_reset=True, use_cupy=True, step_mode='s'):
        class Conv_Base(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation = 1):
                super(Conv_Base, self).__init__()
                if snn_reset:
                    self.conv = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same", dilation=dilation)
                    self.norn = layer.BatchNorm2d(num_features=out_channels)
                else:
                    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same", dilation=dilation)
                    self.norn = nn.BatchNorm2d(num_features=out_channels)
                # self.ac = relu
                # functional.set_step_mode(self, step_mode='m')
                # if use_cupy:
                #     functional.set_backend(self, backend='cupy')

            def forward(self, inputs):
                x = self.conv(inputs)
                x = self.norn(x)
                # x = self.ac(x)

                return x

        super(EBC, self).__init__()
        self.step_mode = step_mode
        # t = channels[0][0]
        if snn_reset:
            self.conv_0 = layer.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, padding="same")
            self.ac = neuron.IFNode(surrogate_function=surrogate.ATan())
        else:
            self.conv_0 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, padding="same")
            self.ac = nn.ReLU()
        self.conv_1 = Conv_Base(channels[1], channels[1] // 2, kernel_size=3)
        self.conv_2 = Conv_Base(channels[1], channels[1] - (channels[1]//2), kernel_size=3, dilation=2)

        # functional.set_step_mode(self, step_mode='m')
        # if use_cupy:
        #     functional.set_backend(self, backend='cupy')


    def forward(self, inputs):
        x = self.conv_0(inputs)
        x0 = self.conv_1(x)
        x1 = self.conv_2(x)
        # x2 = self.ac(torch.cat(torch.cat(x,x0), x1))
        return self.ac(torch.concat([x0,x1], dim=1 if self.step_mode=='s' else 2))

# channels = [[32,64]]
# x = torch.rand(size=(4,32,256,256))
# model = EBC(channels)
# y = model(x)

# print()