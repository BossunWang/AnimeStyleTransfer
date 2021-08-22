#CUDA_VISIBLE_DEVICES=0 python test.py \
#    --input_dir '../../WhiteBoxAnimeGAN/test_img' \
#    --output_dir 'generate_img/' \
#    --checkpoint 'train_20210801/checkpoint/ASMStyleGAN_Epoch_34000.pt' \
#    --resize 2400
#
#CUDA_VISIBLE_DEVICES=0 python test.py \
#    --input_dir '../../collection_photo' \
#    --output_dir 'generate_img/' \
#    --checkpoint 'train_20210801/checkpoint/ASMStyleGAN_Epoch_34000.pt' \
#    --resize 2400

CUDA_VISIBLE_DEVICES=0 python test.py \
    --input_dir 'tmp_img/' \
    --output_dir 'generate_img/' \
    --checkpoint 'train_20210822/checkpoint/ASMStyleGAN_Epoch_59999.pt' \
    --resize 2400

CUDA_VISIBLE_DEVICES=0 python test.py \
    --input_dir 'generate_img/' \
    --output_dir 'generate_img_again/' \
    --checkpoint 'train_20210822/checkpoint/ASMStyleGAN_Epoch_59999.pt' \
    --resize 2400

#python test.py \
#    --input_dir '../../custom_pictures/' \
#    --output_dir 'generate_img/' \
#    --checkpoint 'train_20210809/checkpoint/ASMStyleGAN_Epoch_40000.pt' \
#    --resize 2400
#
#python test.py \
#    --input_dir 'generate_img/' \
#    --output_dir 'generate_img_again/' \
#    --checkpoint 'train_20210809/checkpoint/ASMStyleGAN_Epoch_40000.pt' \
#    --resize 2400
