import torch
from torchvision import datasets, transforms
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from model import VGG19
from sklearn.cluster import KMeans
from shutil import copyfile


def load_image(image_path, rw, rh):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (rw, rh))

    img = torch.from_numpy(img)
    img = img / 127.5 - 1.0
    return img


def main():
    batch_size = 64
    data_dir = '../style_dataset'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg_model = 'vgg19-dcbb9e9d.pth'
    VGG = VGG19(init_weights=vgg_model, feature_mode=True).to(device)
    for param in VGG.parameters():
        param.require_grad = False

    train_features = torch.tensor([]).cpu()
    features = torch.tensor([]).cpu()
    image_path_list = []

    # with torch.no_grad():
    #     for dirPath, dirNames, fileNames in os.walk(data_dir):
    #         for i, f in tqdm(enumerate(fileNames)):
    #             if 'smooth' in dirPath:
    #                 img_path = os.path.join(dirPath, f)
    #                 image = load_image(img_path, 128, 128)
    #                 input = image.permute(2, 0, 1).unsqueeze(0).to(device)
    #                 feature = VGG(input)
    #                 feature = feature.cpu()
    #                 if i < 500:
    #                     train_features = torch.cat((train_features, feature.view(1, -1)), 0)
    #                 features = torch.cat((features, feature.view(1, -1)), 0)
    #                 image_path_list.append(img_path)
    #
    #     features = features.cpu().detach().numpy()
    #     np.save('train_features.npy', train_features)
    #     np.save('features.npy', features)
    #     np.save('image_path_list.npy', image_path_list)

    train_features = np.load('train_features.npy')
    features = np.load('features.npy')
    # print(features.shape)
    # print(image_path_list)

    reducer = umap.UMAP()
    embedding = reducer.fit(train_features)

    # plt.scatter(
    #     embedding[:, 0],
    #     embedding[:, 1])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Anime dataset', fontsize=24)
    # plt.show()

    embedding = reducer.transform(features)
    class_number = 4
    kmeans = KMeans(n_clusters=class_number, random_state=0).fit(embedding)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=list(kmeans.labels_))
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Anime dataset', fontsize=24)
    plt.savefig('UMAP_Kmeans_clusters.jpg')

    image_path_list = np.load('image_path_list.npy')

    labe_list = np.arange(class_number)
    for label in labe_list:
        os.makedirs(os.path.join(data_dir, 'cluster_labels', str(label), 'style'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'cluster_labels', str(label), 'smooth'), exist_ok=True)

    for index, image_path in enumerate(image_path_list):
        label = kmeans.labels_[index]
        image_file = image_path.split('/')[-1]
        smooth_image_path = image_path
        style_image_path = image_path.replace('smooth', 'style')

        new_smooth_image_path = os.path.join(data_dir, 'cluster_labels', str(label), 'smooth', image_file)
        new_style_image_path = os.path.join(data_dir, 'cluster_labels', str(label), 'style', image_file)
        copyfile(smooth_image_path, new_smooth_image_path)
        copyfile(style_image_path, new_style_image_path)


if __name__ == '__main__':
    main()