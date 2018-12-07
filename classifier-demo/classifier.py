
# coding: utf-8

# # Training a Classifier
# 
# We have seen how to implement one simple classifier. Now let's see how to combine classifiers into a Neural Network. Pytorch has classes that allow us to build Neural Networks without verbose code. Those classes are in the torch.nn module

# ### Loading Data
# For this classifier, we will use a dataset called CIFAR10 which contains 10000 images of 32x32 pixels, correcly classified within ten different categories.

import demo.data as data

print('Possible categories')
print(data.classes())

# Get the training data
train_data = data.training_data()

# Print a sample batch of 4 pictures, with the labels
data.sample_batch(train_data)


# ### Training the network
# Now we can iterate over our training data a specified amount of times and make sure that our Network gets better at understanding it. In this case we are iterating over the data twice.

import demo.network as neuralnet

# Create our Neural Network
net = neuralnet.NeuralNetwork()

# Train in our training dataset
neuralnet.train(net, train_data, 2)


# ### Checking the output 
# After our Network has been trained, it means it now has parameters that it can use to analyze any image. These parameters are just like the weights we've seen in our previous example. But instead of having just one dimension, in this case, they have 32x32x3 dimensions initially and instead of having just one result they have an output of 10 possible results: 
# 

params = list(net.parameters())
print(params)
print(len(params))


# So now it's time to test our parameters against our initial set of data. We can use our recently trained network in the images from the dataset and see how it performs against the ground truth labels:

import demo.test as tester

# Get our testing dataset
test_data = data.testing_data()
classes = data.classes()

# Test our neural net in one small sample (4 pictures)
(img, truth, prediction) = tester.test_neuralnet_sample(net, test_data, classes)

# Show our results
data.imshow(img)
print('Ground Truth ', truth)
print('Prediction ', prediction)


# And we can also run a report of the efficiency of our predictions against the whole dataset

tester.test_neuralnet_all(net, test_data, classes)

