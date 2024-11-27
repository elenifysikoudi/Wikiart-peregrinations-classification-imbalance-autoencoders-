import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
import torchvision.transforms as transforms
import numpy as np


''' 
A file that contains the dataset and models used.
''' 

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
])

class WikiArtImage:

    '''
    A class to represent an image from the WikiArt dataset.

    Attributes:
    - imgdir (str): The directory where the image is stored.
    - label (str): The class or label of the image.
    - filename (str): The filename of the image.
    - image (Tensor or None): A Tensor representation of the image (loaded once accessed).
    - loaded (bool): A flag indicating whether the image has been loaded.
    - transform (callable): A transformation to be applied to the image, if any.

    Methods:
    - get(): Loads and returns the image from disk, applying transformations if necessary.
    '''
    def __init__(self, imgdir, label, filename):

        '''
        Arguments:
        - imgdir (str): The directory where the image is stored.
        - label (str): The label or class of the image.
        - filename (str): The name of the image file.
        '''
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False
        self.transform = transform

    def get(self):

        '''
        Loads the image from the specified file path and applies transformations if defined.

        Returns:
        - Tensor: The image loaded as a PyTorch tensor. The image is loaded only once when accessed.
        '''
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            #if self.transform:
                #self.image = self.transform(self.image)
            self.loaded = True

        return self.image
        

class WikiArtDataset(Dataset):

    '''
    A custom dataset class for the WikiArt dataset.

    Attributes:
    - imgdir (str): The directory where the images are stored.
    - classes (list): The list of class labels.
    - device (str): The device to load the tensors onto.
    - filedict (dict): A dictionary mapping image filenames to corresponding WikiArtImage objects.
    - indices (list): A list of all image filenames in the dataset.
    - sample_weights (list): The sample weights to be used for imbalanced class handling.

    Methods:
    - _calculate_sample_weights(): Calculates the sample weights for imbalanced class distribution.
    - __len__(): Returns the number of images in the dataset.
    - __getitem__(idx): Returns the image and its corresponding label at the given index.
    '''
    
    def __init__(self, imgdir, classes ,device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                indices.append(art)
        print("...finished")
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = classes
        self.device = device

        self.sample_weights = self._calculate_sample_weights()

    def _calculate_sample_weights(self):

        '''
        Calculates sample weights to handle class imbalance.

        This function calculates the weight for each sample in the dataset. It assigns higher weights 
        to the samples of underrepresented classes to balance the class distribution.

        Returns:
        - sample_weights (list): A list of sample weights for each image in the dataset.
        '''
        
        class_counts = [0] * len(self.classes)
        for idx in range(len(self)):
            _, ilabel = self[idx]
            class_counts[ilabel] += 1

        class_weights = 1. / np.array(class_counts)

        sample_weights = [class_weights[self.classes.index(self.filedict[img].label)] for img in self.indices]
        
        return sample_weights
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        return image, ilabel

class WikiArtModel(nn.Module):

    '''
    A convolutional neural network (CNN) model designed for image classification, specifically for classifying art styles.

    Arguments:
    -num_classes(int) : The number of classes (art styles) to classify. Default is 27.

    Returns:
        The log probabilities of the predicted class for each input image in the batch(torch.Tensor).
    '''
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv2d = nn.Conv2d(3, 16, (4,4), padding=2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool2d = nn.MaxPool2d((4,4), padding=2)
        self.conv2d_2 = nn.Conv2d(16, 32, (4,4), padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2d_2 = nn.MaxPool2d((4, 4), padding=2)
        self.conv2d_3 = nn.Conv2d(32,64,(4,4) , padding = 2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 8 * 8, 300)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        #print("convout {}".format(output.size()))
        output = self.batchnorm1(output)
        output = self.relu(output)
        output = self.maxpool2d(output)
        #print("poolout {}".format(output.size())) 
        output = self.conv2d_2(output)  
        #print("conv2 {}".format(output.size())) 
        output = self.batchnorm2(output)
        output = self.relu(output)
        output = self.maxpool2d_2(output)
        #print("poolout2 {}".format(output.size())) 
        output = self.conv2d_3(output)
        output = self.batchnorm3(output)
        output = self.maxpool2d_2(output)
        #print("conv3 {}".format(output.size())) 
        output = self.flatten(output)
        #print("flatten {}".format(output.size()))        
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return self.softmax(output)

def gather_classes(root_dir):

    ''' Collects the unique classes from the dataset directory and sorts them.

        Arguments:
        - root_dir (str): the directory containing the classes of the data.

        Returns:
            A sorted list of unique class names to be used for the dataset.
    '''
    
    classes = set()
    for root, _, files in os.walk(root_dir):
        arttype = os.path.basename(root)
        if files:  
            classes.add(arttype)
    return sorted(classes)

def calculate_class_distribution(dataset):

    '''
    Calculates the distribution of classes in the given dataset.

    Arguments:
    - dataset (WikiArtDataset): The dataset object containing images and labels. 
      It is expected to have a `classes` attribute which is a list of all the labels.
      and a `filedict` attribute which is a dictionary where keys are file names and values are WikiArtImage objects with labels.
    Returns:
    - dict: A dictionary where the keys are class labels and the values are the number of samples/images for each class.
   ''' 
    
    class_counts = {}
    for cls in dataset.classes:
        class_counts[cls] = 0  
    for art_obj in dataset.filedict.values():
        if art_obj.label in class_counts:
            class_counts[art_obj.label] += 1
    return class_counts

class Autoencoder(nn.Module):
    '''
    A convolutional autoencoder model that learns to encode and decode images.

    Arguments:
    -latent_dim(int): The dimensionality of the latent space in the autoencoder, which is the output size of the encoder and the input size for the decoder.
        Default: 300.

    Returns:
    A tuple containing:
        - "encoded" (torch.Tensor): The latent representation of the input image.
        - "decoded" (torch.Tensor): The reconstructed image.
    '''
    def __init__(self,latent_dim =300):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 52 * 52, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 52 * 52),  
            nn.ReLU(),
            nn.Unflatten(1, (128, 52, 52)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



class ArtStyleEmbedding(nn.Module):

    '''
    A simple neural network module that creates embeddings for art styles.

    Arguments:
    -num_styles(int): The number of styles in the dataset.
    -embedding_dim(int): The dimensionality of the embedding space,which determines the size of each art style's embedding vector.
    '''
    def __init__(self, num_styles, embedding_dim):
        super(ArtStyleEmbedding, self).__init__()
        self.style_embedding = nn.Embedding(num_styles, embedding_dim)  

    def forward(self, style_idx):
        return self.style_embedding(style_idx)

class ConditionalAutoencoder(nn.Module):

    '''
    A conditional autoencoder that reconstructs images based on both image features and style embeddings.

    This model takes an image and a style index as input. The image is passed through an encoder to extract features,
    while the style index is mapped to a dense vector through a style embedding layer. These features are then combined
    and passed to the decoder to reconstruct the original image.

    Arguments:
    -image_encoder_decoder(nn.Module): A model containing both the encoder and decoder components. The encoder extracts features from input images,
        and the decoder reconstructs images from the combined latent and style features.
    -style_embedding_dim(int): The dimensionality of the style embeddings.

    Returns:
    The reconstructed image tensor after passing through the decoder(torch.Tensor).
    '''
    def __init__(self, image_encoder_decoder, style_embedding_dim):
        super(ConditionalAutoencoder, self).__init__()
        self.encoder = image_encoder_decoder.encoder
        self.style_embedding = ArtStyleEmbedding(num_styles=27, embedding_dim=style_embedding_dim)
        self.decoder = image_encoder_decoder.decoder

        latent_dim = 300
        self.latent_dim = latent_dim
        flattened_size = 64 * 52 * 52

        self.decoder[0] = nn.Linear(latent_dim + style_embedding_dim, flattened_size)

    def forward(self, image, style_idx):
        image_features = self.encoder(image)  
        style_features = self.style_embedding(style_idx) 
        
        combined_features = torch.cat([image_features, style_features], dim=-1)
        
        reconstructed_image = self.decoder(combined_features)
        return reconstructed_image
