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
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()


def make_edge_smooth(dataset_name, img_size, target_dir='smooth'):
    check_folder(os.path.join(dataset_name, target_dir))

    file_list = glob(os.path.join(dataset_name, 'style', "*.*"))
    save_dir = os.path.join(dataset_name, target_dir)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list) :
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # make_edge_smooth('../../AnimeGANv2/dataset/new_anime_dataset/Hayao', args.img_size)
    # make_edge_smooth('../../AnimeGANv2/dataset/new_anime_dataset/Paprika', args.img_size)
    # make_edge_smooth('../../AnimeGANv2/dataset/new_anime_dataset/Shinkai', args.img_size)
    # make_edge_smooth('../style_dataset/CG', args.img_size)
    # make_edge_smooth('../style_dataset/safebooru', args.img_size)

    make_edge_smooth('../style_dataset/CG', args.img_size * 2, 'smooth_HD')
    make_edge_smooth('../style_dataset/safebooru', args.img_size * 2, 'smooth_HD')


if __name__ == '__main__':
    main()
