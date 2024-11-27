import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, gather_classes, Autoencoder
import json
import argparse
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]
epochs=config["epochs"]
batch_size=config['batch_size']

print("Running...")


all_classes = gather_classes(trainingdir)


def save_and_cluster_representations(autoencoder, dataset, device="cuda:0", n_clusters=27, reduction_method="pca",modelfile=None):
    """
    Save and cluster representations from the autoencoder encoder, then visualize.
    
    Arguments:
        -autoencoder: Trained autoencoder model.
        -dataset: Dataset object (should provide images and labels).
        -device: Device to perform computation on.
        -n_clusters: Number of clusters to form.
        -reduction_method: Dimensionality reduction method ("pca", "tsne").
        - modelfile (str): The path to save the trained model's state_dict.
    """

    model = autoencoder
    model.load_state_dict(torch.load(modelfile, weights_only=True))
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    autoencoder.eval()  
    autoencoder.to(device)
    
    all_representations = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader, desc="Extracting representations", unit="batch"):
            images = images.to(device)
            encoded, _ = autoencoder(images)  
            flattened_representations = encoded.view(encoded.size(0), -1)  
            all_representations.append(flattened_representations.cpu().numpy())
            all_labels.extend(labels.numpy())

    representations = np.vstack(all_representations)
    all_labels = np.array(all_labels)

    
    clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(representations)
    cluster_labels_predict = clustering.predict(representations)

    
    if reduction_method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    elif reduction_method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    
    reduced_representations = reducer.fit_transform(representations)

    class_names = dataset.classes

    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(
        reduced_representations[:, 0], reduced_representations[:, 1],
        c=cluster_labels_predict, cmap='gist_rainbow', alpha=0.6
    )
    plt.colorbar(scatter, label="Cluster Label")

    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        label_mask = all_labels == label
        label_representations = reduced_representations[label_mask]
        class_name = class_names[label]  
        plt.scatter(
            label_representations[:, 0], 
            label_representations[:, 1], 
            label=class_name, alpha=0.6
        )

    plt.title("2D Visualization of Clusters by True Labels (Encoded Representations)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.savefig("cluster_visualization_true_labels_with_text.png")
    plt.show()

    return reduced_representations, cluster_labels_predict

autoencoder = Autoencoder()
testdataset = WikiArtDataset(testingdir, classes=all_classes, device=device)

reduced_representations, cluster_labels = save_and_cluster_representations(
    autoencoder=autoencoder, 
    dataset=testdataset, 
    device="cuda:0",  
    n_clusters=27,  
    reduction_method="tsne",
    modelfile="autoencoder.pth"
)