# TEC - Tomorrow's Experts in Computing
## Deep Learning: How computers learn to recognize pictures

This project provides a code base for TEC workshop 'Deep Learning: How computers learn to recognize pictures'.

The main files in the project just contain the construct of a typical machine learning project but the essential part are deleted. The project contains several facilities to load, save and visualize data.

In the practical part of the TEC workshop, paticipants will download this project. Together with the workshop leaders the code will be completed. The participants are entitled encouraged to improve the code and play around to get insight into the world wor machine learning.

## Structure
- dataset // Empty folder used by PyTorch to download and process datasets 
- images // Empty folder for images of handwritten images to classify (used by 'playground.py')
- model // Folder where model and training courves are save to / loaded form
- tec // root packages
  - dnn // package for handwritten digit recognition using neuronal networks on MNIST
    - architecture.py // Collection of architectures to train (TODO complete code and add new architectures)
    - playground.py // Loads a model and does some visualizations
    - training.py // Code for training a neural network (TODO complete code)
  - kmeans // package for a simple K-means algorithm
    - kmean.py // Performa a k-means algorithm on some data from gen_data.py (TODO complete code)
  - utils // collection of provided utils
    - gen_data.py // Generates random datasets for the k-means algorithm
    - plotting.py // Contains some wrapper classes that simplify vizualizing data and training process with matplotlib
    - torch_utils.py // Helps to store and load PyTorch model to stop and continue the training process
    
