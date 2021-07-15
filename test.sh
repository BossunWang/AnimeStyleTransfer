CUDA_VISIBLE_DEVICES=0 python test.py \
    --input_dir '../../WhiteBoxAnimeGAN/test_img' \
    --output_dir 'generate_img/' \
    --checkpoint 'checkpoint_20210628/AnimeGAN_Epoch_150.pt' \
    --resize_scalar 4
