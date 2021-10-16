import torch
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans
from shutil import copyfile
from VGG import VGGFeatures
import pickle
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN


def load_image(image_path, rw, rh, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.resize(img, (rw, rh))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = (img / 255.0 - mean) / std
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))
    return img


def extract_feature(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_plan = {0: device}
    pooling = 'average'
    style_layers = [1, 6, 11, 20, 29]
    model = VGGFeatures(style_layers, pooling=pooling)
    model.distribute_layers(device_plan)

    data_size_list = []
    features = torch.tensor([]).cpu()
    image_path_list = []

    with torch.no_grad():
        for dirPath, dirNames, fileNames in os.walk(data_dir):
            if 'smooth' == dirPath.split('/')[-1]:
                data_size_list.append(len(fileNames))
                for i, f in tqdm(enumerate(fileNames)):
                    img_path = os.path.join(dirPath, f)
                    image = load_image(img_path, 256, 256)
                    input = image.unsqueeze(0).to(device)
                    feature_dict = model(input)
                    feature = feature_dict[29].cpu()
                    features = torch.cat((features, feature.view(1, -1)), 0)
                    image_path_list.append(img_path)

        features = features.cpu().detach().numpy()
        np.save('features.npy', features)
        with open('image_path_list.pickle', 'wb') as f:
            pickle.dump(image_path_list, f)
        np.save('data_size_list.npy', np.array(data_size_list))


def UMAP_embedding():
    data_size_list = np.load('data_size_list.npy')
    features = np.load('features.npy')
    print('features size:', features.shape)

    train_features = []
    start = 0
    end = 0
    for data_size in data_size_list:
        end += data_size
        part_features = features[start:end]
        part_index = np.random.choice(data_size, 1000)
        train_features.extend(part_features[part_index])
        start += data_size

    train_features = np.array(train_features)
    reducer = umap.UMAP()
    mapper = reducer.fit(train_features)

    # Training with Labels and Embedding Unlabelled Test Data (Metric Learning with UMAP)
    embedding = mapper.transform(features)
    np.save('embedding.npy', embedding)

    plt.scatter(*embedding.T, s=1.0, cmap='Spectral', alpha=1.0)
    plt.title('UMAP projection of the Anime dataset', fontsize=24)
    plt.savefig('UMAP_metric_learning.jpg')


def cluster(data_dir):
    embedding = np.load('embedding.npy')
    with open('image_path_list.pickle', 'rb') as f:
        image_path_list = pickle.load(f)
        image_path_list = np.array(image_path_list)

    # exclude outlier
    class_number = 2
    clustering = SpectralClustering(n_clusters=class_number
                                    , assign_labels='discretize'
                                    , affinity='nearest_neighbors'
                                    , random_state=0).fit(embedding)
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=3.0, cmap='Spectral', alpha=1.0,
        c=list(clustering.labels_))
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(class_number + 1) - 0.5)
    cbar.set_ticks(np.arange(class_number))
    plt.title('UMAP projection of the Anime dataset', fontsize=24)
    plt.savefig('UMAP_SpectralClustering.jpg')

    # cluster again exclude center cluster
    second_embedding = embedding[clustering.labels_ == 0]
    second_clustering = DBSCAN(eps=0.5, min_samples=2).fit(second_embedding)
    # print(np.max(second_clustering.labels_))
    # print(np.min(second_clustering.labels_))
    plt.cla()
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(
        second_embedding[:, 0],
        second_embedding[:, 1],
        s=3.0, cmap='Spectral', alpha=1.0,
        c=list(second_clustering.labels_))
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(np.min(second_clustering.labels_), np.max(second_clustering.labels_) + 1) - 0.5)
    cbar.set_ticks(np.arange(np.min(second_clustering.labels_), np.max(second_clustering.labels_)))
    plt.title('UMAP projection of the Anime dataset', fontsize=24)
    plt.savefig('UMAP_DBSCAN.jpg')

    max_cluster_index = 0
    max_cluster_size = 0
    for index in range(np.min(second_clustering.labels_), np.max(second_clustering.labels_)):
        if len(second_clustering.labels_[second_clustering.labels_ == index]) > max_cluster_size:
            max_cluster_size = len(second_clustering.labels_[second_clustering.labels_ == index])
            max_cluster_index = index

    third_embedding = second_embedding[[second_clustering.labels_ == max_cluster_index]]
    print("third_embedding size", third_embedding.shape)
    plt.cla()
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(
        third_embedding[:, 0],
        third_embedding[:, 1],
        s=3.0, cmap='Spectral', alpha=1.0)
    plt.title('UMAP projection of the Anime dataset', fontsize=24)
    plt.savefig('UMAP_DBSCAN_exclude_outlier.jpg')

    image_path_list0 = image_path_list[clustering.labels_ == 0]
    image_path_list1 = image_path_list[clustering.labels_ == 1]
    image_path_list0_exclude_outlier = image_path_list0[[second_clustering.labels_ == max_cluster_index]]
    image_path_list0_outlier = image_path_list0[[second_clustering.labels_ != max_cluster_index]]
    embedding0 = second_embedding[[second_clustering.labels_ == max_cluster_index]]
    embedding1 = embedding[clustering.labels_ == 1]
    label0 = np.zeros(embedding0.shape[0])
    label1 = np.ones(embedding1.shape[0])

    final_embedding = np.concatenate([embedding0, embedding1])
    final_labels = np.concatenate([label0, label1])
    final_image_path_list = np.concatenate([image_path_list0_exclude_outlier, image_path_list1])

    print('image_path_list0_exclude_outlier:', image_path_list0_exclude_outlier.shape)
    print('image_path_list1:', image_path_list1.shape)
    print('embedding0:', embedding0.shape)
    print('embedding1:', embedding1.shape)
    print('final_image_path_list:', final_image_path_list.shape)

    plt.cla()
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(
        final_embedding[:, 0],
        final_embedding[:, 1],
        c=list(final_labels),
        s=3.0, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(class_number + 1) - 0.5)
    cbar.set_ticks(np.arange(class_number))
    plt.title('UMAP projection of the Anime dataset', fontsize=24)
    plt.savefig('UMAP_final_cluster.jpg')

    labe_list = np.arange(class_number)
    for label in labe_list:
        os.makedirs(os.path.join(data_dir, 'cluster_labels', str(label), 'style'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'cluster_labels', str(label), 'smooth'), exist_ok=True)

    os.makedirs('outlier', exist_ok=True)

    for index, image_path in enumerate(final_image_path_list):
        label = int(final_labels[index])
        image_file = image_path.split('/')[-1]
        smooth_image_path = image_path
        style_image_path = image_path.replace('smooth', 'style')

        new_smooth_image_path = os.path.join(data_dir, 'cluster_labels', str(label), 'smooth', image_file)
        new_style_image_path = os.path.join(data_dir, 'cluster_labels', str(label), 'style', image_file)
        copyfile(smooth_image_path, new_smooth_image_path)
        copyfile(style_image_path, new_style_image_path)

    for index, image_path in enumerate(image_path_list0_outlier):
        image_file = image_path.split('/')[-1]
        style_image_path = image_path.replace('smooth', 'style')
        new_style_image_path = os.path.join('outlier', image_file)
        copyfile(style_image_path, new_style_image_path)


def main():
    data_dir = '../style_dataset'
    extract_feature(data_dir)
    UMAP_embedding()
    cluster(data_dir)


if __name__ == '__main__':
    main()
