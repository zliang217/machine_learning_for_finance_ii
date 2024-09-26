"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
import mnist_loader 

# Third-party libraries
from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    print("Data loaded. Fitting now.")
    # train
    clf = svm.SVC(verbose=True)
    clf.fit(training_data[0], training_data[1])
    print("Fitting complete. Predicting now.")
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    print("Prediction complete. Calculating accuracy now.")
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("This is a baseline classifier using an SVM.")
    print("{} of {} values are correct.".format(num_correct, len(test_data[1])))

if __name__ == "__main__":
    svm_baseline()
    