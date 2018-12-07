# Pytorch Demos

This is a set of PyTorch demos that I initially prepared for the Developer Circle Leads Summit in Lagos, December 2018

I expect to be able to grow this repository of demos 

## Prerequisits
- Install [Anaconda](https://conda.io/docs/user-guide/install/macos.html)
- Install [pyTorch](https://pytorch.org/get-started/locally/)
- Install [Jupyter](http://jupyter.org/install)
- Additional packages to install (use `conda install`)
  - `torchvision`
  - `numpy`
  - `matplotlib`

## Cat Demo
This shows how to train a simple perceptron to answer the question "Can we estimate if a Facebook post contains a cat picture based solely on the amounts of likes and shares"?
The dataset does not contain any real data and it's a simple artificial clustered dataset that simulates that the more people share and like a post, the more likely it is that it has a cat picutre.

## Neural Network Demo
All the pyTorch complexities are hidden away in helper modules, which allows for demoing a Neural Network being trained and classifying, without getting hung up on details. As a follow up to the presentation, we can show the audience the inner workings of the helper modules.



