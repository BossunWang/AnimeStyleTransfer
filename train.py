"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from AverageMeter import AverageMeter

from train_dataset import SourceImageDataset, TargetImageDataset
from components.Conditional_Generator_asm import Generator
from components.Conditional_Discriminator_Projection import Discriminator
from components.Transform import Transform_block
import loss
import matplotlib.pyplot as plt


def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(train_loader_src, train_loader_tgt
                    , generator, discriminator, Transform
                    , G_init_optimizer, G_optimizer, D_optimizer
                    , reconstruct_loss_meter
                    , discriminator_loss_meter, generator_loss_meter
                    , content_loss_meter, style_loss_meter, color_loss_meter, tv_loss_meter, transform_loss_meter
                    , cur_epoch, conf, update_G_only=False):
    j = conf.training_rate
    train_loader_tgt_iterator = iter(train_loader_tgt)

    for batch_idx, (x, x_gray) in enumerate(train_loader_src):
        batch_size = x.size(0)
        training_data_size = len(train_loader_src)

        try:
            y, y_gray, y_smooth, y_smooth_gray, labels = next(train_loader_tgt_iterator)
        except StopIteration:
            train_loader_tgt_iterator = iter(train_loader_tgt)
            y, y_gray, y_smooth, y_smooth_gray, labels = next(train_loader_tgt_iterator)

        if update_G_only:
            global_epoch = cur_epoch
        else:
            global_epoch = cur_epoch - conf.init_epoch

        global_batch_idx = global_epoch * training_data_size + batch_idx
        x = x.to(conf.device)
        x_gray = x_gray.to(conf.device)
        y = y.to(conf.device)
        y_gray = y_gray.to(conf.device)
        y_smooth = y_smooth.to(conf.device)
        y_smooth_gray = y_smooth_gray.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.long()

        if update_G_only:
            G_x, G_x_feature = generator(x, labels)
            reconstruct_loss = loss.con_loss_func(x, G_x)

            # update G
            G_init_optimizer.zero_grad()
            reconstruct_loss.backward()
            G_init_optimizer.step()

            reconstruct_loss_meter.update(reconstruct_loss.item(), batch_size)
            if batch_idx % conf.print_freq == 0:
                reconstruct_loss_val = reconstruct_loss_meter.avg
                lr = get_lr(G_init_optimizer)
                print('Epoch %d, iter %d, lr %f, reconstruct loss %f' %
                      (cur_epoch, batch_idx, lr, reconstruct_loss_val))
                conf.writer.add_scalar('reconstruct_loss', reconstruct_loss_val, global_batch_idx)
                conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
                reconstruct_loss_meter.reset()
        else:
            # training times , G : D = 1 : self.training_rate
            # Update D
            G_x, G_x_feature = generator(x, labels)
            D_real = discriminator(y, labels)
            D_photo = discriminator(x, labels)
            D_gray = discriminator(y_gray, labels)
            D_blur = discriminator(y_smooth_gray, labels)
            D_fake = discriminator(G_x, labels)
            Discriminator_loss = conf.d_adv_weight * \
                                 loss.discriminator_hinge_loss_func(D_real, D_photo, D_gray, D_blur, D_fake)

            D_optimizer.zero_grad()
            Discriminator_loss.backward()
            D_optimizer.step()

            if j == conf.training_rate:
                # Update G
                G_x, x_feature = generator(x, labels)
                G_x_feature = generator(G_x, get_feature=True)
                y_gray_feature = generator(y_gray, get_feature=True)
                D_fake = discriminator(G_x, labels)

                generator_loss = conf.g_adv_weight * loss.generator_hinge_loss_func(D_fake)
                content_loss = conf.con_weight * loss.con_loss_func(x_feature, G_x_feature)
                style_loss = conf.sty_weight * loss.style_loss_func(y_gray_feature, G_x_feature)
                color_loss = conf.color_weight * loss.color_loss_func(x, G_x)
                tv_loss = conf.tv_weight * loss.total_variation_loss(G_x)
                transform_loss = conf.transform_weight * loss.transform_loss(Transform(x), Transform(G_x))

                Generator_loss = generator_loss + content_loss + style_loss + color_loss + tv_loss + transform_loss

                G_optimizer.zero_grad()
                Generator_loss.backward()
                G_optimizer.step()

            discriminator_loss_meter.update(Discriminator_loss.item(), batch_size)
            generator_loss_meter.update(generator_loss.item(), batch_size)
            content_loss_meter.update(content_loss.item(), batch_size)
            style_loss_meter.update(style_loss.item(), batch_size)
            color_loss_meter.update(color_loss.item(), batch_size)
            tv_loss_meter.update(tv_loss.item(), batch_size)
            transform_loss_meter.update(transform_loss.item(), batch_size)

            if batch_idx % conf.print_freq == 0:
                discriminator_loss_val = discriminator_loss_meter.avg
                generator_loss_val = generator_loss_meter.avg
                content_loss_val = content_loss_meter.avg
                style_loss_val = style_loss_meter.avg
                color_loss_val = color_loss_meter.avg
                tv_loss_val = tv_loss_meter.avg
                transform_loss_val = transform_loss_meter.avg

                lr = get_lr(G_init_optimizer)
                print('Epoch %d, iter %d, lr %f'
                      ', discriminator loss %f'
                      ', generator loss %f'
                      ', content loss %f'
                      ', style loss %f'
                      ', color loss %f'
                      ', tv loss %f'
                      ', transform_loss_val %f'
                      % (cur_epoch, batch_idx, lr
                         , discriminator_loss_val
                         , generator_loss_val
                         , content_loss_val
                         , style_loss_val
                         , color_loss_val
                         , tv_loss_val
                         , transform_loss_val))

                conf.writer.add_scalar('discriminator_loss', discriminator_loss_val, global_batch_idx)
                conf.writer.add_scalar('generator_loss', generator_loss_val, global_batch_idx)
                conf.writer.add_scalar('content_loss', content_loss_val, global_batch_idx)
                conf.writer.add_scalar('style_loss', style_loss_val, global_batch_idx)
                conf.writer.add_scalar('color_loss', color_loss_val, global_batch_idx)
                conf.writer.add_scalar('tv_loss', tv_loss_val, global_batch_idx)
                conf.writer.add_scalar('transform_loss', transform_loss_val, global_batch_idx)

                conf.writer.add_scalar('Train_lr', lr, global_batch_idx)

                discriminator_loss_meter.reset()
                generator_loss_meter.reset()
                content_loss_meter.reset()
                style_loss_meter.reset()
                color_loss_meter.reset()
                tv_loss_meter.reset()
                transform_loss_meter.reset()

            j = j - 1
            if j < 1:
                j = conf.training_rate

    if cur_epoch % conf.save_freq == 0 or cur_epoch == conf.epoch - 1:
        saved_name = 'AnimeGAN_Epoch_%d.pt' % cur_epoch
        if update_G_only:
            state = {
                'epoch': cur_epoch
                , 'generator': generator.module.state_dict()
            }
        else:
            state = {
                'epoch': cur_epoch
                , 'generator': generator.module.state_dict()
                , 'discriminator': discriminator.module.state_dict()
                , 'G_optimizer': G_optimizer.state_dict()
                , 'D_optimizer': D_optimizer.state_dict()
            }

        torch.save(state, os.path.join(conf.checkpoint_dir, saved_name))
        print('save checkpoint %s to disk...' % saved_name)


def test(test_loader_src, generator, class_num, cur_epoch, conf):
    for n, (x, x_gray) in enumerate(test_loader_src):
        x = x.to(conf.device)
        labels = np.arange(class_num)
        labels = labels.reshape(1, -1).repeat(x.size(0), axis=0)
        labels = torch.from_numpy(labels).to(conf.device)
        labels = labels.long()

        for i in range(class_num):
            G_recon, _ = generator(x, labels[:, i].view(-1))
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(conf.result_dir,
                                str(cur_epoch) + '_epoch_' + 'test_' + str(n + 1) + "_label_" + str(i) + '.png')
            plt.imsave(path, (result.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
        if n == 4:
            break


def train(conf):
    """Total training procedure. 
    """
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = tuple(conf.img_size)
    train_data_src = SourceImageDataset('../../AnimeGANv2/dataset/train_photo/', input_size=img_size)
    train_data_tgt = TargetImageDataset('../../AnimeGANv2/dataset/new_anime_dataset', input_size=img_size)
    test_data_tgt = SourceImageDataset('../../AnimeGANv2/dataset/test/test_photo/', input_size=img_size)

    train_loader_src = torch.utils.data.DataLoader(train_data_src
                                                   , batch_size=conf.batch_size
                                                   , shuffle=True
                                                   , drop_last=True)
    train_loader_tgt = torch.utils.data.DataLoader(train_data_tgt
                                                   , batch_size=conf.batch_size
                                                   , shuffle=True
                                                   , drop_last=True)
    test_loader_src = torch.utils.data.DataLoader(test_data_tgt
                                                  , batch_size=1
                                                  , shuffle=True)

    generator = Generator(class_num=train_data_tgt.get_class_size()).to(conf.device)
    discriminator = Discriminator(n_class=train_data_tgt.get_class_size()).to(conf.device)
    Transform = Transform_block().to(conf.device)

    # Adam optimizer
    G_init_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                    generator.parameters()), lr=conf.g_lr, betas=(0.5, 0.999))
    G_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          generator.parameters()), lr=conf.g_lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          discriminator.parameters()), lr=conf.d_lr, betas=(0.5, 0.999))

    ori_epoch = 0

    if conf.pretrained:
        checkpoint = torch.load(conf.pretrain_model, map_location=conf.device)
        if "epoch" in checkpoint \
                and "generator" in checkpoint \
                and "discriminator" in checkpoint \
                and "G_optimizer" in checkpoint \
                and "D_optimizer" in checkpoint:
            print('load general model')
            ori_epoch = checkpoint['epoch'] + 1
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        # load reconstruct model
        elif "epoch" in checkpoint \
                and "generator" in checkpoint:
            print('load reconstruct model')
            ori_epoch = checkpoint['epoch'] + 1
            generator.load_state_dict(checkpoint['generator'])

    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)

    generator.train()
    discriminator.train()

    reconstruct_loss_meter = AverageMeter()
    discriminator_loss_meter = AverageMeter()
    generator_loss_meter = AverageMeter()
    content_loss_meter = AverageMeter()
    style_loss_meter = AverageMeter()
    color_loss_meter = AverageMeter()
    tv_loss_meter = AverageMeter()
    transform_loss_meter = AverageMeter()

    print('start epoch:', ori_epoch)
    for epoch in range(ori_epoch, conf.epoch):
        if epoch < conf.init_epoch:
            train_one_epoch(train_loader_src, train_loader_tgt
                            , generator, discriminator, Transform
                            , G_init_optimizer, G_optimizer, D_optimizer
                            , reconstruct_loss_meter
                            , discriminator_loss_meter, generator_loss_meter
                            , content_loss_meter, style_loss_meter, color_loss_meter, tv_loss_meter, transform_loss_meter
                            , epoch, conf, update_G_only=True)
        else:
            train_one_epoch(train_loader_src, train_loader_tgt
                            , generator, discriminator, Transform
                            , G_init_optimizer, G_optimizer, D_optimizer
                            , reconstruct_loss_meter
                            , discriminator_loss_meter, generator_loss_meter
                            , content_loss_meter, style_loss_meter, color_loss_meter, tv_loss_meter, transform_loss_meter
                            , epoch, conf, update_G_only=False)

        test(test_loader_src, generator, train_data_tgt.get_class_size(), epoch, conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='semi-siamese_training for face recognition.')
    parser.add_argument('--src_dataset', type=str, default='train_photo', help='dataset_name')
    parser.add_argument('--tgt_dataset', type=str, default='Hayao', help='dataset_name')
    parser.add_argument('--val_dataset', type=str, default='val', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=10, help='The number of epochs for weight initialization')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='The size of batch size')  # if light : batch_size = 20
    parser.add_argument('--print_freq', type=int, default=100, help='The number of loss print freq')
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--init_lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--g_lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=4e-5, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--g_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--d_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--color_weight', type=float, default=15.,
                        help='Weight about color loss')  # 15. for Hayao, 50. for Paprika, 10. for Shinkai
    parser.add_argument('--tv_weight', type=float, default=1.,
                        help='Weight about tv')  # 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
    parser.add_argument('--con_weight', type=float, default=1.2,
                        help='Weight about feature')
    parser.add_argument('--sty_weight', type=float, default=1.,
                        help='Weight about style loss')  # 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
    parser.add_argument('--transform_weight', type=float, default=50.,
                        help='Weight about Transform block')  # 1. for Hayao, 0.1 for Paprika, 1. for Shinkai

    # ---------------------------------------------
    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D')
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')

    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H and W')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--vgg_model', type=str, default='vgg19-dcbb9e9d.pth',
                        help='file name to load the vgg model for feature extraction')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='whether to pretrained')
    parser.add_argument('--pretrain_model', type=str, default='checkpoint/',
                        help='file name to load the model for training')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    tensorboardx_logdir = os.path.join(args.log_dir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    train(args)
