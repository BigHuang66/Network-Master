# coding='utf-8'
"""t-SNE 样本分布 进行可视化"""
import cv2
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.manifold import TSNE


''' mnist data'''
# def get_data():
#     digits = datasets.load_digits(n_class=6)
#     data = digits.data
#     label = digits.target
#     n_samples, n_features = data.shape
#     return data, label, n_samples, n_features

''' General Data '''
def get_data(input_path):
    img_names = os.listdir(input_path)
    # 创建空间
    data = np.zeros((len(img_names), 128*128))
    label = np.zeros((len(img_names), ), dtype=int)

    label_map = {'USH': 0, 'UFH': 1, 'USQ': 2, 'UFQ': 3, 'DFQ': 4, 'DSQ': 5, 'DFH': 6, 'DSH': 7}
    for i in range(len(img_names)):
        # feature
        img_path = os.path.join(input_path, img_names[i])
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (128, 128))
        img = img.reshape(1, -1)
        data[i] = img

        # label
        label[i] = label_map[img_names[i].split('.')[0]]
    
    n_samples, n_features = data.shape

    return data, label, n_samples, n_features


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i]+1) / 10),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def plot_embedding_3D(data,label,title): 
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0) 
    data = (data- x_min) / (x_max - x_min) 
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    for i in range(data.shape[0]): 
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9}) 
    return fig


def main():
    data, label, n_samples, n_features = get_data('./datasets/val')
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding_2D(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    main()