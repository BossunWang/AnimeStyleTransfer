import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from components.Conditional_ResBlock import Conditional_ResBlock
from components.ASM import ASM


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        
        pad_layer = {
            "zero":    nn.ZeroPad2d,
            "same":    nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError
            
        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch*expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        
        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class ConvSpectralNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        super(ConvSpectralNorm, self).__init__()

        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        layers = [pad_layer[pad_mode](padding), spectral_norm(nn.Conv2d(in_ch
                                                                        , out_ch
                                                                        , kernel_size=kernel_size
                                                                        , stride=stride
                                                                        , padding=0
                                                                        , groups=groups
                                                                        , bias=bias))]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        return out

    
class Generator(nn.Module):
    def __init__(self, chn=32, k_size=3, class_num=3):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  chn, kernel_size=7, padding=3),
            # padding_left padding_right padding_top  padding_bottom
            ConvNormLReLU(chn, chn * 2, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(chn * 2, chn * 2)
        )
        
        self.block_b = nn.Sequential(
            ConvNormLReLU(chn * 2,  chn * 4, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(chn * 4, chn * 4)
        )
        
        self.block_c = nn.Sequential(
            ConvNormLReLU(chn * 4, chn * 4),
            InvertedResBlock(chn * 4, chn * 8, 2),
            InvertedResBlock(chn * 8, chn * 8, 2),
            InvertedResBlock(chn * 8, chn * 8, 2),
            InvertedResBlock(chn * 8, chn * 8, 2),
            ConvNormLReLU(chn * 8, chn * 8)
        )

        self.conditional_res = Conditional_ResBlock(chn * 8, k_size, class_num)
        self.asm = ASM(in_channels=chn * 8, out_channel=chn * 2, attr_dim=class_num, up_scale=2, norm='in')
        
        self.block_d = nn.Sequential(
            ConvNormLReLU(chn * 8, chn * 4),
            ConvNormLReLU(chn * 4, chn * 2)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(chn * 4, chn * 2),
            ConvNormLReLU(chn * 2,  chn * 2),
            ConvNormLReLU(chn * 2,  chn, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(chn, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

        self.__weights_init__()

    def __weights_init__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # try:
                #     nn.init.zeros_(m.bias)
                # except:
                #     print("No bias found!")

            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input, condition=None, align_corners=True, get_feature=False):
        out_a = self.block_a(input)
        out_b = self.block_b(out_a)

        if get_feature:
            return out_b

        out_c = self.block_c(out_b)
        out = self.conditional_res(out_c, condition)

        # print('out_a:', out_a.size())
        # print('out:', out.size())
        output_ASM = self.asm(out_a, out, condition)

        # about align_corners: https://zhuanlan.zhihu.com/p/87572724
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=align_corners)
        out = self.block_d(out)

        concat_out = torch.cat((out, output_ASM), dim=1)

        out = F.interpolate(concat_out, scale_factor=2, mode="bilinear", align_corners=align_corners)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, chn=32, k_size=3, n_class=3):
        super(Discriminator, self).__init__()

        enable_bias = True

        # stage 1
        self.conv1 = nn.Sequential(
            ConvSpectralNorm(3, chn, stride=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.aux_classfier1 = nn.Sequential(
            ConvSpectralNorm(chn, chn, kernel_size=5, bias=enable_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.embed1 = spectral_norm(nn.Embedding(n_class, chn))
        self.linear1 = spectral_norm(nn.Linear(chn, 1))

        # stage 2
        self.conv2 = nn.Sequential(
            ConvSpectralNorm(chn, chn * 2, stride=2, pad_mode='zero'),
            nn.LeakyReLU(0.2, inplace=True),
            ConvSpectralNorm(chn * 2, chn * 4, stride=1, pad_mode='zero'),
            nn.GroupNorm(num_groups=1, num_channels=chn * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.aux_classfier2 = nn.Sequential(
            ConvSpectralNorm(chn * 4, chn, kernel_size=5, bias=enable_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed2 = spectral_norm(nn.Embedding(n_class, chn))
        self.linear2 = spectral_norm(nn.Linear(chn, 1))

        # stage 3
        self.conv3 = nn.Sequential(
            ConvSpectralNorm(chn * 4, chn * 4, stride=2, pad_mode='zero'),
            nn.LeakyReLU(0.2, inplace=True),
            ConvSpectralNorm(chn * 4, chn * 8, stride=1, pad_mode='zero'),
            nn.GroupNorm(num_groups=1, num_channels=chn * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.aux_classfier3 = nn.Sequential(
            ConvSpectralNorm(chn * 8, chn, kernel_size=5, bias=enable_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed3 = spectral_norm(nn.Embedding(n_class, chn))
        self.linear3 = spectral_norm(nn.Linear(chn, 1))

        self.conv = nn.Sequential(
            ConvSpectralNorm(3, 32, stride=1),
            nn.LeakyReLU(0.2, inplace=True),

            ConvSpectralNorm(32, 64, stride=2, pad_mode='zero'),
            nn.LeakyReLU(0.2, inplace=True),
            ConvSpectralNorm(64, 128, stride=1, pad_mode='zero'),
            nn.GroupNorm(num_groups=1, num_channels=128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            ConvSpectralNorm(128, 128, stride=2, pad_mode='zero'),
            nn.LeakyReLU(0.2, inplace=True),
            ConvSpectralNorm(128, 256, stride=1, pad_mode='zero'),
            nn.GroupNorm(num_groups=1, num_channels=256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            ConvSpectralNorm(256, 1, stride=1, pad_mode='zero')
        )

        self.__weights_init__()

    def __weights_init__(self):
        # print("Init weights")
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # try:
                #     nn.init.zeros_(m.bias)
                # except:
                #     print("No bias found!")

            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input, condition):
        h = self.conv1(input)
        prep1 = self.aux_classfier1(h)
        # print('prep1:', prep1.size())
        prep1 = prep1.view(prep1.size()[0], -1)
        y1 = self.embed1(condition)
        # print('y1:', y1.size())
        y1 = torch.sum(y1 * prep1, dim=1, keepdim=True)
        # print('y1:', y1.size())
        prep1 = self.linear1(prep1) + y1

        h = self.conv2(h)
        prep2 = self.aux_classfier2(h)
        prep2 = prep2.view(prep2.size()[0], -1)
        y2 = self.embed2(condition)
        y2 = torch.sum(y2 * prep2, dim=1, keepdim=True)
        prep2 = self.linear2(prep2) + y2

        h = self.conv3(h)
        prep3 = self.aux_classfier3(h)
        prep3 = prep3.view(prep3.size()[0], -1)
        y3 = self.embed3(condition)
        y3 = torch.sum(y3 * prep3, dim=1, keepdim=True)
        prep3 = self.linear3(prep3) + y3

        out_prep = [prep1, prep2, prep3]
        return out_prep


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    class_num = 3
    generator = Generator(class_num=3).to(device)
    generator.eval()

    input = torch.rand(batch_size, 3, 256, 256).to(device)
    label = torch.randint(0, class_num - 1, (batch_size,)).to(device)
    label = label.long()

    upsample_align = True
    output = generator(input, label, upsample_align)
    print('output:', output.size())

    discriminator = Discriminator().to(device)
    discriminator.eval()

    prep1, prep2, prep3 = discriminator(input, label)
    print(prep1.size())
    print(prep2.size())
    print(prep3.size())