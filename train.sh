CUDA_VISIBLE_DEVICES=1,0 python train.py \
    --src_dataset '../Photo_image_training_data/' \
    --tgt_dataset '../style_dataset/cluster_labels' \
    --val_dataset '../../places365/val/val_256' \
    --reconstruct_iter 5000 \
    --max_iter 40000 \
    --print_freq 100 \
    --save_freq 1000 \
    --test_freq 500 \
    --d_adv_weight 1. \
    --g_adv_weight 3. \
    --con_weight 2. \
    --sty_weight 1. \
    --color_weight 10. \
    --tv_weight 1. \
    --transform_weight 50. \
    --d_steps_per_iter 1 \
    --g_steps_per_iter 1 \
    --mixed_precision True
#    --pretrained True \
#    --pretrain_model 'checkpoint/ASMStyleGAN_Epoch_4000.pt'

# fine-tune by HD
#CUDA_VISIBLE_DEVICES=0,1 python train.py \
#    --src_dataset '../../DIV2K_HR/crop_HD' \
#    --tgt_dataset '../style_dataset/cluster_labels' \
#    --tgt_style_dir 'style_HD' \
#    --tgt_smooth_dir 'smooth_HD' \
#    --val_dataset '../../DIV2K_HR/crop_HD' \
#    --img_size 512 \
#    --batch_size 1 \
#    --reconstruct_iter 5000 \
#    --max_iter 60000 \
#    --print_freq 300 \
#    --save_freq 1000 \
#    --test_freq 500 \
#    --d_adv_weight 1. \
#    --g_adv_weight 5. \
#    --con_weight 2. \
#    --sty_weight 1. \
#    --color_weight 10. \
#    --tv_weight 1. \
#    --transform_weight 50. \
#    --training_rate 1 \
#    --pretrained True \
#    --pretrain_model 'train_20210809/checkpoint/ASMStyleGAN_Epoch_40000.pt'
