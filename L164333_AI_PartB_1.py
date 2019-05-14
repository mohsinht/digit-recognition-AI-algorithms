import math
from mnist import MNIST
import numpy as np
from random import randint

mndata = MNIST('./')
readimages, readlabels = mndata.load_training()
readtimages, readtlabels = mndata.load_testing()

images = np.asarray(readimages)
labels = np.asarray(readlabels)
timages = np.asarray(readtimages)
tlabels = np.asarray(readtlabels)

images = images/255
timages = timages/255

print("Loading!")

weights = np.zeros((10, 784))   # 10 arrays of size 28*28 containing 0's. Corresponding weights for each digit.
T = np.zeros((10, 1))           # 10 arrays of size 1 containing 0's. Threshold.
B = 0                   # Bias
S = 0.1                 # Step or Learning_Coefficient
num_of_cycles = 10     # number of cycles for learning


def classify(obj, i, threshold=0):
    if (np.dot(obj, weights[i]) + B) >= threshold:
        return 1
    else:
        return 0


def learn(digit):
    for i in range(0, num_of_cycles):
        for img, lab in zip(images, labels):
            org = 0
            if lab == digit:
                org = 1
            pred = classify(img, digit, T[digit])     #predicted: 0 if false, 1 if true
            dif = org - pred
            weights[digit] = weights[digit] + ((S * dif) * np.asarray(img))
            T[digit] -= (S * dif)

            # print("Threshold " + str(i) + " => " + str(T[digit]))


print("Learning Started!")

for i in range(0, 10):
    print("Learning for Digit " + str(i))
    learn(i)

print("Learning Complete!")


def validate_perceptron(total_checks):
    correct_count = 0
    for h in range(0, total_checks):
        TEST_INDEX = randint(0, 9999)  # randomly choose a test image
        orig = tlabels[TEST_INDEX]
        pred = -1
        for m in range(0, 10):
            if classify(timages[TEST_INDEX], m, T[m][0]) == 1:
                pred = m
                break
        if pred == orig:
            correct_count += 1
        print("Original: " + str(orig) + "  =>  Predicted: " + str(pred))
    print(str(correct_count) + " are right out of " + str(total_checks))
    return correct_count


validate_perceptron(100)
