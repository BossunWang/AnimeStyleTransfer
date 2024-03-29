import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from functools import partial


class Scale(nn.Module):
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'(scale): {self.scale.item():g}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale

# "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                       std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = models.vgg19(pretrained=True).features[:self.layers[-1] + 1]
        self.devices = [torch.device('cpu')] * len(self.model)

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]: # before 'M'
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices):
        for i, layer in enumerate(self.model):
            if i in devices:
                device = devices[i]
            self.model[i] = layer.to(device)
            self.devices[i] = device

    def forward(self, input, layers=None):
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        # input = self.normalize(input)
        for i in range(max(layers) + 1):
            input = self.model[i](input.to(self.devices[i]))
            if i in layers:
                feats[i] = input
        return feats


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_plan = {0: device}
    pooling = 'average'
    style_layers = [1, 6, 11, 20, 29]
    model = VGGFeatures(style_layers, pooling=pooling)
    model.distribute_layers(device_plan)

    batch_size = 4
    input = torch.rand(batch_size, 3, 256, 256).to(device)
    features = model(input, style_layers)

    for style_layer in style_layers:
        print(features[style_layer].size())