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
from wikiart import WikiArtDataset, gather_classes, WikiArtModel
import torcheval.metrics as metrics
import json
import argparse

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

def test(modelfile=None, device="cpu"):

    '''
    Evaluates the performance of the model.

    Arguments:
    -modelfile(str or None): The path to save the trained model's state_dict.
    -device(str):Device to perform computation on, the default is "cpu".

    Returns:
    None
        Prints accuracy and confusion matrix to the console.
    '''
    loader = DataLoader(testingdataset, batch_size=1)

    model = WikiArtModel()
    model.load_state_dict(torch.load(modelfile, weights_only=True))
    model = model.to(device)
    model.eval()

    predictions = []
    truth = []
    for batch_id, batch in enumerate(tqdm.tqdm(loader)):
        X, y = batch
        y = y.to(device)
        output = model(X)
        predictions.append(torch.argmax(output).unsqueeze(dim=0))
        truth.append(y)

    #print("predictions {}".format(predictions))
    #print("truth {}".format(truth))
    predictions = torch.concat(predictions)
    truth = torch.concat(truth)
    metric = metrics.MulticlassAccuracy()
    metric.update(predictions, truth)
    print("Accuracy: {}".format(metric.compute()))
    confusion = metrics.MulticlassConfusionMatrix(27)
    confusion.update(predictions, truth)
    print("Confusion Matrix\n{}".format(confusion.compute()))
    
test(modelfile=config["modelfile"], device=device)