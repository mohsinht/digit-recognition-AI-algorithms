import numpy as np
from mnist import MNIST
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import time

start = time.time();

print('Loading Data ...');

mndata = MNIST('./')
readimages, readlabels = mndata.load_training()
readtimages, readtlabels = mndata.load_testing()


X_train = np.asarray(readimages)
Y_train = np.asarray(readlabels)
X_test = np.asarray(readtimages)
Y_test = np.asarray(readtlabels)


print('Data Load Complete');
X_train = X_train[1:501, :];
Y_train = Y_train[1:501];
X_test = X_test[1:51,:];
Y_test = Y_test[1:51];
X_train = X_train/255;
X_test = X_test/255;
print('Calculation started');


def calculate_hog_features(images):
    list_hog_fd = [hog(t.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                       visualise=False) for t in images]

    return np.array(list_hog_fd, dtype=np.float64)


def run_svm_experiment(train_images, train_labels, test_images, test_labels):
    s = "SVM with HOG based feature descriptor"

    print("------------------------------------------------------------------------------------")
    print("Running Experiment using %s" % s)
    print("------------------------------------------------------------------------------------")

    print("Calculating HOG based Feature Descriptor")
    start_time = time.clock()
    train_hog_features = calculate_hog_features(train_images)
    print("Feature Descriptor calculated !")

    print("Training ...")
    clf = LinearSVC()
    clf.fit(train_hog_features, train_labels)
    print("Training Time: %.2f seconds" % (time.clock() - start_time))
    print("Training Done!")

    print("Classifying Test Images ...")
    start_time = time.clock()
    test_hog_features = calculate_hog_features(test_images)
    predicted_labels = clf.predict(test_hog_features)
    print("Prediction Time: %.2f seconds" % (time.clock() - start_time))

    print("Test Images Classified!")
    accuracy = accuracy_score(test_labels, predicted_labels) * 100

    print("Accuracy: %f" % accuracy, "%")
    print("---------------------\n")


""" run svm using HOG as feature descriptor """
run_svm_experiment(X_train, Y_train, X_test, Y_test);