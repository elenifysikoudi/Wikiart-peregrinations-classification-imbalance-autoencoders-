# Wikiart-peregrinations-classification-imbalance-autoencoders-

The files don't have any arguments that are needed to be run as they are available in the config or hardcoded. To run them all is needed is python3 file.py

The files with the saved models for part 2 and part 3 were too big for github.

# Bonus A

files: ```train.py, test.py```

The initial accuracy was around ``` 0.02 ```

I noticed that there was an error in the way of making the datasets. That is, the train and test dataset had the classes
in different sequence so for instance Impressionsism could have been style 5 in train and style 10 in test that was causing a problem
when testing. To fix that I made a function called gather classes which is called in the train.py and test.py that sorts the styles so they are in the same order. Some other changes that I made were:
-Changing the learning rate from ```0.1 to 0.001```.
-Added batch normalization depending on the number of out features(16,32,64) and maxpooling after every convolutional layer.
-Added a dropout layer to prevent overfitting ```0.3```.
With these changes the accuracy of the model raised to:
```0.29 so around 30%```

# Part 1

files: ```train.py, test.py```

To fix the imbalance of the dataset I tried to different methods.

```1st Method: Weighted Sampling```

I calculated the class distribution with a function and assigned sample_weights to each class based on their frequency in the dataset.WeightedRandomSampler is used to sample images from the WikiArtDataset during training. The sampler is provided with the calculated sample weights based on class distribution, ensuring that underrepresented classes are over-sampled in each batch.

The results of this method weren't good. The accuracy actually fell to ```0.26``` which I think happened because the imbalance is too high for some art styles so the model got noise instead of more valuable data.

```2nd Method: SMOTE upsampling```

I used the SMOTE algorithm to generate syntethetic examples for the underrepresented classes. I didn't use a CNN for the feature extraction I chose the simplest way to do it by just flattening the images. Then, used the calculate_class_distrubution and created a sampling_strategy as auto was taking very long time to run. If the class had less than 320 samples the SMOTE will augment that class by adding another 250 samples. I tried adding more but it seemed like the model was overfitting.

The results of this method was a bit better but not huge difference. The accuracy went up to ```0.31 so 31%```. I believe if I had used a CNN to get image features the results would have been even better.

The upsampling method is hardcoded in the train function as SMOTE because it had better results but it can be changed inside the train.py file.

# Part 2

1. 
file : ```train_encoder.py```

For the autoencoder I created a new nn.Module which consists of an encoder and a decoder.

The encoder consists of 3 convolutional layers leading up to 128 features(32,64,128). Each convolutional layer is proceeded by ReLU activation. After it is flattened into 1D vector to be passed through a fully connected linear layer to get the latent representation.

The decoder consists of a linear layer that transform the latent representation back to a dense vector of the previous shape(128 channels, 52x52 spatial dimensions). Then 3 deconvolutional layers were used to transform the tensor(image) back to its original size followed by a ReLU activation again. 

To evaluate how the encoder was learning I just looked at the loss which was gradually going down as shown in the file "encoder.png".

2. 
file : ```clusters.py```

For the clustering I tried both TSNE and PCA methods. They both had similar results but from my observation TSNE worked better. The clustering plot is available in the file "cluster_visualization_true_labels_with_text.png". This only contains the TSNE clustering as it was better. However, from my observation after training for 15 epochs I wouldn't say that the art styles are clustering super well. There are some cases where the classes are visibly close but not for all images. In general, they seem to be all over the place. Perhaps more training or adding more layers would make the clustering better.


# Part 3

files : ```train_augmented_encoder.py , test_augmented_encoder.py ```

For the augmented encoder I used the autoencoder from before that I incorporated in another class. I also made another class named ```ArtStyleEmbedding``` which is a simple linear layer that learns an embedding for each art style(class). The way the model works is that every batch passes through the encoder first and the latent representation is concatenated with the style embedding and finally it is passed through the decoder to reconstruct the image. 
The training seemed to go well since the model looked like it was learning.
After testing with mismatched styles specifically it randomly chooses one of the art styles, the image that comes back which is in the file ```reconstructed_image``` along with the original is just noise. It's pixels connected but showing nothing. Perhaps more training would make it look better but for sure it wouldn't give the results of a GAN.


