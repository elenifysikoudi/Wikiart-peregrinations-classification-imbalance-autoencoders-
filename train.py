import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler,TensorDataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, gather_classes, WikiArtModel, calculate_class_distribution
import json
import argparse
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

print("Running...")


all_classes = gather_classes(trainingdir)

traindataset = WikiArtDataset(trainingdir, classes=all_classes, device=device)
#testingdataset = WikiArtDataset(testingdir, device)

print(traindataset.imgdir)

the_image, the_label = traindataset[5]
print(the_image, the_image.size())

# the_showable_image = F.to_pil_image(the_image)
# print("Label of img 5 is {}".format(the_label))
# the_showable_image.show()

sampler = WeightedRandomSampler(traindataset.sample_weights, num_samples=len(traindataset), replacement=True)

def train(epochs=3, batch_size=32, modelfile=None, device="cuda:0", upsampling=None):
    '''
    Trains the WikiArtModel for the specified number of epochs.

    Arguments:
    - epochs (int): The number of epochs to train the model. Default is 3.
    - batch_size (int): The batch size used for training. Default is 32.
    - modelfile (str): The path to save the trained model's state_dict.
    - device (str): The device to perform computations on. Default is "cuda:0".
    - upsampling (str or None): Method of upsampling. The options are "weights" and "smote". Default is None.

    Returns:
    - model (WikiArtModel): The trained model.
    '''

    # Weighted sampling
    if upsampling == "weights":
        loader = DataLoader(traindataset, sampler=sampler, batch_size=batch_size)

    # Using SMOTE to upsample
    elif upsampling == "smote":
        X, y = [], []
        for img, label in traindataset:
            X.append(img.reshape(-1).cpu().numpy())
            y.append(label)
        X = np.array(X)
        y = np.array(y)

        class_to_idx = {cls: idx for idx, cls in enumerate(traindataset.classes)}
        class_counts = calculate_class_distribution(traindataset)

        sampling_strategy = {}
        for cls, count in class_counts.items():
            int_cls = class_to_idx[cls]  
            if count > 320:
                sampling_strategy[int_cls] = count
            else:
                sampling_strategy[int_cls] = count + 250

        X = X.reshape(X.shape[0], -1)
        print("Starting SMOTE resampling...")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_resampled = torch.tensor(X_resampled, dtype=torch.float32)
        X_resampled = X_resampled.view(-1, 3, 416, 416)
        y_resampled = torch.tensor(y_resampled, dtype=torch.long)
        resampled_dataset = TensorDataset(X_resampled, y_resampled)

        resampled_class_counts = dict(Counter(y_resampled.numpy()))
        print("Resampled class distribution:", resampled_class_counts)

        loader = DataLoader(resampled_dataset, batch_size=batch_size, shuffle=True)

    # No upsampling
    else:
        loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss().to(device)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss.item()
            optimizer.step()

        print(f"In epoch {epoch}, loss = {accumulate_loss}")

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model


model = train(config["epochs"], config["batch_size"], modelfile=config["modelfile"], device=device,upsampling="smote")
