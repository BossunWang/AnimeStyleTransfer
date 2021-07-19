import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F


def calc_gradient_penalty(netD, real_data, fake_data, gp_weight):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(real_data.device)
    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(real_data.device)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(real_data.device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


def discriminator_loss_func(real, gray, fake, real_blur):
    ones = torch.ones_like(real).to(real.device)
    real_loss = torch.mean((real - ones) ** 2)
    gray_loss = torch.mean(gray ** 2)
    fake_loss = torch.mean(fake ** 2)
    real_blur_loss = torch.mean(real_blur ** 2)

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0
    # lsgan_loss = 1.7 * real_loss + 1.7 * fake_loss + 1.7 * gray_loss + 1.0 * real_blur_loss
    lsgan_loss = 1.0 * real_loss + 1.0 * fake_loss + 1.0 * gray_loss + 1.0 * real_blur_loss
    return lsgan_loss


def discriminator_hinge_loss_func(real, gray, smooth, fake):
    d_loss_real = 0
    for i in range(len(real)):
        temp = Hinge_loss(1 - real[i]).mean()
        # temp *= prep_weights[i]
        d_loss_real += temp

    d_loss_gray = 0
    for i in range(len(gray)):
        temp = Hinge_loss(1 + gray[i]).mean()
        # temp *= prep_weights[i]
        d_loss_gray += temp

    d_loss_smooth = 0
    for i in range(len(smooth)):
        temp = Hinge_loss(1 + smooth[i]).mean()
        # temp *= prep_weights[i]
        d_loss_smooth += temp

    d_loss_fake = 0
    for i in range(len(fake)):
        temp = Hinge_loss(1 + fake[i]).mean()
        # temp *= prep_weights[i]
        d_loss_fake += temp

    d_loss = d_loss_real + d_loss_gray + d_loss_smooth + d_loss_fake
    return d_loss


def generator_loss_func(fake):
    ones = torch.ones_like(fake).to(fake.device)
    lsgan_loss = torch.mean((fake - ones) ** 2)
    return lsgan_loss


def generator_hinge_loss_func(fake):
    g_loss_fake = 0
    for i in range(len(fake)):
        temp = -fake[i].mean()
        # temp *= prep_weights[i]
        g_loss_fake += temp

    return g_loss_fake


def Huber_loss(x, y):
    return F.smooth_l1_loss(x, y)


def Hinge_loss(x):
    return F.relu(x)


def con_loss_func(input_feature, target_feature):
    return F.l1_loss(input_feature, target_feature)


def transform_loss(transform_feature1, transform_feature2):
    return F.mse_loss(transform_feature1, transform_feature2)


def gram_matrix(input):
    a, b, c, d = input.size()
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def style_loss_func(input_feature, target_feature):
    source_gram_matrix = gram_matrix(input_feature)
    target_gram_matrix = gram_matrix(target_feature)
    return F.l1_loss(source_gram_matrix, target_gram_matrix)


# https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out


def color_loss_func(con, fake):
    con_yuv = rgb_to_yuv(con)
    fake_yuv = rgb_to_yuv(fake)

    color_loss = F.l1_loss(con_yuv[:, 0, :, :], fake_yuv[:, 0, :, :]) \
                 + Huber_loss(con_yuv[:, 1, :, :], fake_yuv[:, 1, :, :]) \
                 + Huber_loss(con_yuv[:, 2, :, :], fake_yuv[:, 2, :, :])

    return color_loss


def total_variation_loss(image):
    tv_h = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    tv_loss = (tv_h + tv_w)
    return tv_loss


if __name__ == '__main__':
    from model import Generator, Discriminator, VGG19

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    VGG = VGG19(init_weights='vgg19-dcbb9e9d.pth', feature_mode=True).to(device)
    for param in VGG.parameters():
        param.require_grad = False

    inputs = torch.rand(1, 3, 256, 256).to(device)
    inputs_gray = torch.rand(1, 3, 256, 256).to(device)
    inputs_blur = torch.rand(1, 3, 256, 256).to(device)
    inputs.requires_grad = True
    inputs_gray.requires_grad = True
    inputs_blur.requires_grad = True

    inputs_fake = generator(inputs)
    D_fake = discriminator(inputs_fake)

    D_real = discriminator(inputs)
    D_gray = discriminator(inputs_gray)
    D_blur = discriminator(inputs_blur)
    discriminator_loss = discriminator_loss_func(D_real, D_gray, D_fake, D_blur)
    discriminator_loss.backward()

    inputs_fake = generator(inputs)
    D_fake = discriminator(inputs_fake)

    inputs_feature = VGG(inputs)
    generator_loss = generator_loss_func(D_fake)
    content_loss = con_loss_func(inputs_feature, inputs_feature)
    style_loss = style_loss_func(inputs_feature, inputs_feature)
    color_loss = color_loss_func(inputs, inputs)
    tv_loss = total_variation_loss(inputs)

    total_loss = generator_loss + content_loss + style_loss + color_loss + tv_loss
    total_loss.backward()
    print(total_loss)
