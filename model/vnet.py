
import torch
import torch.nn as nn

def create_stage(conv_num, in_channels, out_channels):
    conv_list = []
    conv = None
    for i in range(conv_num):
        if i == 0:
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        else:
            conv = nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        conv_list.append(conv)
    return nn.Sequential(*conv_list)

create_stage(3, 15, 12)

class Vnet(nn.Module):
    def __init__(self):
        return