# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth
import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def parse_args():
    desc = "Cropped image"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()


def make_edge_smooth(dataset_name, img_size):
    check_folder(os.path.join(dataset_name, 'style'))

    file_list = glob(os.path.join(dataset_name, 'origin', "*.*"))
    save_dir = os.path.join(dataset_name, 'style')

    for f in tqdm(file_list) :
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)

        if bgr_img.shape[0] > bgr_img.shape[1]:
            crop_size = bgr_img.shape[1]
        else:
            crop_size = bgr_img.shape[0]

        crop_img = bgr_img[0:crop_size, 0:crop_size]
        crop_img = cv2.resize(crop_img, (img_size, img_size))
        cv2.imwrite(os.path.join(save_dir, file_name), crop_img)


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_edge_smooth('../style_dataset/CG', args.img_size)


if __name__ == '__main__':
    main()
