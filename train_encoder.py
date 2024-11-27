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

traindataset = WikiArtDataset(trainingdir, classes=all_classes, device=device)

def train_autoencoder(autoencoder, dataset=traindataset, epochs=epochs, batch_size=batch_size, learning_rate=0.001, device=device,modelfile=None):

    '''
    Trains an autoencoder model on the provided dataset.

    Arguments:
    - autoencoder(nn.Module): The autoencoder model to train.
    - dataset(Dataset): The dataset to use for training (default: traindataset).
    - epochs(int): The number of epochs to train for (default: value from config).
    - batch_size(int): The batch size for training (default: value from config).
    - learning_rate(float): The learning rate for the optimizer (default: 0.001).
    - device(str): The device to use for training.
    - modelfile (str): The path to save the trained model's state_dict.

    Returns:
    -autoencoder (nn.Module) : The trained autoencoder model.
    '''
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoencoder = Autoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    epoch_losses = []

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        total_loss = 0
        for batch_id, (images,labels) in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")):
            
            images = images.to(device)
            
            encoded, decoded = autoencoder(images)

            loss = criterion(decoded, images)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    if modelfile:
        torch.save(autoencoder.state_dict(), modelfile)

    return autoencoder
    
autoencoder = Autoencoder()
autoencoder = train_autoencoder(autoencoder,traindataset,epochs=epochs,batch_size=batch_size,learning_rate=0.001,device="cuda:0",modelfile="autoencoder.pth")


