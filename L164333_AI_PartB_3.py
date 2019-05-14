import math
from mnist import MNIST
import numpy as np
from random import randint

mndata = MNIST('./')
readimages, readlabels = mndata.load_training()
readtimages, readtlabels = mndata.load_testing()

print("Loading Data!")

images = np.asarray(readimages)
labels = np.asarray(readlabels)
timages = np.asarray(readtimages)
tlabels = np.asarray(readtlabels)

images = images/255
timages = timages/255

weights = np.zeros((10, 785))   # 10 arrays of size 28*28 containing 0's. Corresponding weights for each digit.
num_of_cycles = 10
alpha = 0.01

row = images.shape[0];
col = images.shape[1];


def train():
    global images;
    global labels;
    bias = np.ones((row, 1));
    images = np.append(bias, images, axis=1);
    for digit in range(10):
        print("Learning for " + str(digit) + "...");
        calculated_weight = gradientDescent(labels == digit);
        weights[digit, :] = calculated_weight;


def gradientDescent(original_label):
    start_weights = np.zeros((col+1, 1));
    num_of_images = images.shape[0];
    original_label = original_label[:, np.newaxis];
    for i in range(num_of_cycles):
        for j in range(num_of_images):
            x = images[j, :];
            x = x[:, np.newaxis];
            y = np.tanh(np.dot(start_weights.T, x));

            start_weights = start_weights + (alpha * (((original_label[j] - y) * (1-y**2)) * x))

    return start_weights.T;


def predict(features, trained_weights):
    z = np.dot(features, trained_weights.T)
    return np.tanh(z)


def validate_perceptron(total_checks):
    correct_count = 0
    global timages;
    a = np.ones((timages.shape[0], 1));
    timages = np.append(a, timages, axis=1);
    for h in range(0, total_checks):
        TEST_INDEX = randint(0, timages.shape[0])  # randomly choose a test image
        orig = tlabels[TEST_INDEX]
        pred = -1
        max_value = -1000
        for m in range(0, 10):
            guess = predict(timages[TEST_INDEX], weights[m])
            if guess >= max_value:
                pred = m
                max_value = guess
        if pred == orig:
            correct_count += 1
        print("Original: " + str(orig) + "  =>  Predicted: " + str(pred))
    print(str(correct_count) + " are right out of " + str(total_checks))
    return correct_count


print("Learning Started!")
train()
print("Learning Complete!")
validate_perceptron(1000)