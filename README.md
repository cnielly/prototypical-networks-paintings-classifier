# A Paintings Classifier with Prototypical Networks
An implementation of the "Shazam for paintings" in Pytorch

## Introduction

Art is both peaceful and powerful. Art can convey a strong message with few elements ("A good sketch is better than a long speech" as Napoléon Bonaparte said). On this project, I decided to focus more specifically on painting. Paintings can be encountered everywhere: in people’s homes, in offices, in malls and of course in museums. The pain point I wanted to solve with my project is the following: we lack information when looking at a painting. Who is the artist? What is the title?  When was it painted? What is the message behind it? What is its estimated price? All these questions usually stay unanswered for an amateur like me. In some places, a small cart board gives 2 or 3 key information about the painting, but that’s it. 

So what if I could create a sort of “Shazam for paintings”? In other words, create an algorithm that could recognize a painting from a simple picture taken with a smartphone, and then provide extra information to the user. Imagine: you are in front of a painting that you do not know. Take out your phone, open the app, take a picture and you can instantaneously learn about it. 

In fact, this scenario perfectly fits the FSL framework, for three reasons:
-  **Few examples per class:** pictures of painting are not very common, especially for paintings that are not famous. 
-  **A large number of classes:** Art is infinite. Not only the number of paintings in the world is huge, it will continue rising as artists paint new pieces every day. The system needs to take into account this constant motion. 
-  Painting is a **field that is quite diverse:** from romanticism to cubism, styles vary from a painting to another. This will be interesting because the algorithm will have to find an embedding space that is capable of integrating all kinds of artistic movements. 

This “Shazam for painting” could be very useful to museums, to give an alternative to audio guides and cardboards.

## Create the dataset:

The first step was to create the dataset to train on. A few open source paintings datasets exist on the web, such as the Kaggle dataset “Best Artworks of all time” (16k images, from 50 artists) or the GitHub dataset “WikiArt” (200k paintings from 3k artists). Unfortunately, these are useless for our application. Firstly, because they only have one example per painting, the painting itself in a very high quality (sort of a scan). We need more than one example per class (at least one for the support set and one for the query set). Secondly, because the task we are trying to learn is to recognize a painting from a picture of it. To be consistent, we need a dataset of pictures of paintings. 

Since that dataset does not exist on the web, I decided to create it on my own, for the sake of the experiment. I gathered pictures from two sources: famous paintings, taken from Google Image, and less famous paintings, taken from my home and friend’s places. In the obtained dataset, each class (= painting) contains six examples: one is the purest example I could find (very good picture) and five are pictures taken from various angles, with various qualities and different lightness. This is to reproduce the real-life scenario: pictures people will take from their smartphones won’t be perfect. 

<p align="center">
<img src="https://github.com/cnielly/prototypical-networks-paintings-classifier/blob/master/README_images/6_versions.JPG" width="500" alt="Clusters in the embedding space">
</p>

The obtained paintings dataset can be found on the repository, and is organized as followed:

<p align="center">
<img src="https://github.com/cnielly/prototypical-networks-paintings-classifier/blob/master/README_images/tree_files.JPG" width="500" alt="Clusters in the embedding space">
</p>

After a few hours of work, I was able to gather 30 classes, each containing 6 examples. So a total number of 180 pictures. I split this dataset into a training set of 20 classes and a testing set of 10 classes. 

The next step is to choose an algorithm and code it. In the line of [previous work on the Omniglot dataset](https://github.com/cnielly/prototypical-networks-omniglot), I decided to use Prototypical Networks.

## Code and launch training:

Images are read with the OpenCV Python library. They go through resizing (56x56 pixels). 

Each sample is composed of 1 support image and 5 query images. With this configuration, we are in a One-shot Learning framework. At each episode, a batch of 20 randomly picked samples is used. So, our training parameters are: 

*Nc = 20, Ns = 1, Ns = 5*
  
The sample then goes through four embedding blocks, and each image is transformed from (56x56) to a (64x3x3) tensor, as shown on the image bellow. 

<p align="center">
<img src="https://github.com/cnielly/prototypical-networks-paintings-classifier/blob/master/README_images/embedding_module.JPG" width="500" alt="Clusters in the embedding space">
</p>

Then prototypes are computed for each class thanks to the support set. Here, no average is needed since there is only one image per class (One-shot). The prototype is actually the support image itself. Distances are computed between queries and prototypes (Euclidean), and finally a class is assigned to each of the 5 query images. 
Loss is computed the loss with cross-entropy. And the backpropagate allows the weights of the network to be updated. Training settings are: 

*3 epochs, 200 episodes per epoch, learning rate divided by 2 at each epoch*

## Launch testing:

Once the network is trained, it is time to test. Imagine a museum comes to us with a set of paintings. We do not want to retrain our network from scratch, but we want to test if our network is able to recognize these new classes. 
I tested the network on 1,000 episodes. At each episode a batch of 5 classes is used, as if the museum would have come to us with 5 different paintings (with 6 pictures of each). So our training parameters are: 

*Nc = 5, Ns = 1, Ns = 5 * 

## Results: 

I was very impressed by the accuracy of the network, given the few amount of data I was able to gather in my dataset. I obtain **0.99** accuracy on the testing set.
