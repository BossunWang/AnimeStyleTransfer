import torch
from torch import nn


class Transform_block(nn.Module):
    def __init__(self, k_size = 10):
        super().__init__()
        padding_size = int((k_size -1)/2)
        # self.padding = nn.ReplicationPad2d(padding_size)
        self.pool = nn.AvgPool2d(k_size, stride=1,padding=padding_size)

    def forward(self, input_image):
        # h = self.padding(input)
        out = self.pool(input_image)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = Transform_block().to(device)

    input = torch.rand(8, 256, 16, 16).to(device)
    output = transform(input)
    print(output.size())