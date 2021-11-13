CUDA_VISIBLE_DEVICES=0 python test_interpolation.py \
    --input_dir1 'tmp_img/' \
    --input_dir2 'train_20211031/generate_img_again/' \
    --output_dir 'train_20211031/generate_interpolation_img/' \
    --checkpoint 'train_20211031/checkpoint/ASMStyleGAN_Epoch_39999.pt' \
    --resize 2400

