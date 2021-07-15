CUDA_VISIBLE_DEVICES=1,0 python train.py \
    --src_dataset '../../AnimeGANv2/dataset/train_photo/' \
    --tgt_dataset '../../AnimeGANv2/dataset/new_anime_dataset' \
    --val_dataset '../../AnimeGANv2/dataset/test/test_photo/' \
    --init_epoch 0 \
    --epoch 400 \
    --print_freq 300 \
    --save_freq 2 \
    --d_adv_weight 1. \
    --g_adv_weight 1. \
    --con_weight 50. \
    --sty_weight 1. \
    --color_weight 1. \
    --tv_weight 1. \
    --transform_weight 50. \
    --training_rate 1
#    --pretrained True \
#    --pretrain_model 'checkpoint/AnimeGAN_Epoch_8.pt'