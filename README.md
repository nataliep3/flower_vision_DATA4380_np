![](UTA-DataScience-Logo.png)

# Computer Vision Flower Image Classification

This repository outlines an attempt to use transfer learning models to classify images of flowers using the dataset "Flower Images Dataset" on Kaggle, found [here](https://www.kaggle.com/datasets/aksha05/flower-image-dataset).

## Overview

The task, as defined by the Kaggle challenge is to use a set of 733 images of flowers to train a transfer learning model to accurately classify these images into one of 10 possible classes of flowers. In this repository, four models are attempted using three different transfer learning models (MobileNetv2, ResNet101v2, ResNet50) to attempt to achieve the highest non-trivial accuracy scores. The MobileNetv2 model with five total image augmentations achieved the highest validation accuracy score of 0.91, meaning that 91% of the time, the model was able to accurately classify an image of a flower.

## Summary of Workdone

### Data

* Data:
  * Type: For example
    * Input: Flower images (244x244 pixel jpegs), image filename -> flower type
    * Class names: 10 total; tulips, orchids, peonies, hydrangeas, lilies, gardenias, garden roses, daisies, hibiscus, bougainvillea
  * Size: 733 unique images
  * Instances (Train, Test, Validation Split): 733 images for use, with 587 images for training and 146 images for validation 

#### Preprocessing / Clean up

* To ensure the directory was organized correctly, I manually sorted the flowers into sub-directories named after each class
* For ease of training, I created a data loading module to use for each new notebook

#### Data Visualization

The image below displays as 3x3 grid of images after being augmented (randomly flipped and rotated), used for intial base modeling

![](image_augmentation.png) 

The image below shows a 3x3 grid of images after being augmented even further (randomly flipped, rotated, zoomed, translated, or adjusted contrast), used for model iterations

![](further_augmented_grid.png) 

### Problem Formulation

* Input: randomly augmented images belonging to one of ten possible classes
* Output: model metrics to determine model performance
* Models
  * MobileNetv2: I worked with this model initially to build off of a provided example and establish a decent baseline set of metrics
  * ResNet101v2: I chose this model because it had a high accuracy score on the Keras website
  * ResNet50: I chose this model to work with a different version of ResNet and determine if there are noticable differences in metrics
* Loss: Sparse categorical cross entropy to handle the multiple class options
* Optimizer: Adam
* Metrics: Sparse categorical accuracy

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







