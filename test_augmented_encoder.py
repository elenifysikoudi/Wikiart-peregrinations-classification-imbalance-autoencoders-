import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, gather_classes, Autoencoder, ConditionalAutoencoder
import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

testingdir = config["testingdir"]
trainingdir = config["trainingdir"]
device = config["device"]


print("Running...")

all_classes = gather_classes(trainingdir)
#traindataset = WikiArtDataset(trainingdir, device)
testingdataset = WikiArtDataset(testingdir, classes=all_classes, device=device)


def test_augmented_encoder(modelfile=None, device="cuda:0"):
    '''
    Tests the augmented encoder by displaying an original image and a reconstructed image.

    Arguments:
    -modelfile(str): The path the model is saved.
    -device(str): The device to use for training.

    It randomly chooses an image from the test dataset matched with a random style and saves 
    the plotting in a file to see it.

    '''

    random_idx = random.randint(0, len(testingdataset) - 1)
    image, label = testingdataset[random_idx] 
    
    model =ConditionalAutoencoder(image_encoder_decoder=Autoencoder(latent_dim=300), style_embedding_dim=8).to(device)
    model.load_state_dict(torch.load(modelfile, weights_only=True))
    model.eval()
    
    image = image.unsqueeze(0).to(device)
    style_idx = torch.randint(0, 27, (image.size(0),)).to(device)
    reconstructed_image = model(image,style_idx)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(F.to_pil_image(image.squeeze().cpu()))  
    plt.axis("off")
    plt.show()   
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(F.to_pil_image(reconstructed_image.squeeze().cpu()))  
    plt.title('Reconstructed Image with Mismatched(random) Style')
    plt.savefig("reconstructed_image")
   
    
test_augmented_encoder("augmented_autoencoder.pth",device)