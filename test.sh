CUDA_VISIBLE_DEVICES=0 python test.py \
    --input_dir '../../WhiteBoxAnimeGAN/test_img' \
    --output_dir 'generate_img/' \
    --checkpoint 'checkpoint/AnimeGAN_Epoch_99.pt' \
    --resize_scalar 5

CUDA_VISIBLE_DEVICES=0 python test.py \
    --input_dir '../../collection_photo' \
    --output_dir 'generate_img/' \
    --checkpoint 'checkpoint/AnimeGAN_Epoch_99.pt' \
    --resize_scalar 2
