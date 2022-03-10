import numpy
import sklearn
from part2.featureengineer import get_feature1_train_data, get_feature1_vector, get_feature2_train_data
from part2.processing import X_dev, Y_dev, X_test, Y_test

accuracy, precision, recall, F_score = 0, 0, 0, 0


# function to get linear classifier instance
def train_svm_classifier(X_train, Y_train):
    svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf.fit(numpy.asarray(X_train), numpy.asarray(Y_train))
    return svm_clf


# function for validation
def validation(model, vocabulary, X_dev, Y_dev):
    X_true = []
    Y_true = []

    for i in X_dev.index:
        X_true.append(get_feature1_vector(vocabulary, X_dev.loc[i, "content"]))
        Y_true.append(Y_dev.loc[i])

    print(vocabulary)
    print(X_true)
    predictions = model.predict(X_true)
    Y_true = numpy.asarray(Y_true)

    accuracy = sklearn.metrics.accuracy_score(Y_true, predictions)
    precision = sklearn.metrics.precision_score(Y_true, predictions, average="macro")
    recall = sklearn.metrics.recall_score(Y_true, predictions, average="macro")
    F_score = sklearn.metrics.f1_score(Y_true, predictions, average="macro")

    return accuracy, precision, recall, F_score


# good performance evaluation with accuracy > 65%
# function to use dev set to adjust n to get an acceptable score, which makes accuracy >= 0.65
def adjust_feature1_n():
    n = 50
    vocabulary_feature1 = []
    accuracy = 0
    while accuracy < 0.2:
        X_train_feature1, Y_train_feature1, vocabulary_feature1 = get_feature1_train_data(n)
        svm_model_1 = train_svm_classifier(X_train_feature1, Y_train_feature1)
        accuracy, precision, recall, F_score = validation(svm_model_1, vocabulary_feature1, X_dev, Y_dev)
        n += 1
    print("Accuracy: " + str(accuracy))
    print("macro-averaged precision: " + str(precision))
    print("macro-averaged recall: " + str(recall))
    print("macro-averaged F_score: " + str(F_score))
    print("n: " + str(n))

    return svm_model_1, vocabulary_feature1


# function to use test set to test the overall performance
def test_feature1_performance():
    model_1, vocabulary_1 = adjust_feature1_n()
    accuracy, precision, recall, F_score = validation(model_1, vocabulary_1, X_test, Y_test)
    print("Accuracy: " + str(accuracy))
    print("macro-averaged precision: " + str(precision))
    print("macro-averaged recall: " + str(recall))
    print("macro-averaged F_score: " + str(F_score))


# test_feature1_performance()

def adjust_feature2_n():
    n = 2
    vocabulary_feature2 = []
    accuracy = 0
    while accuracy < 0.65:
        X_train_feature2, Y_train_feature2, vocabulary_feature2 = get_feature2_train_data(n)
        svm_model_2 = train_svm_classifier(X_train_feature2, Y_train_feature2)
        accuracy, precision, recall, F_score = validation(svm_model_2, vocabulary_feature2, X_dev, Y_dev)
        n += 1
    print("Accuracy: " + str(accuracy))
    print("macro-averaged precision: " + str(precision))
    print("macro-averaged recall: " + str(recall))
    print("macro-averaged F_score: " + str(F_score))
    print("n: " + str(n))

    return svm_model_2, vocabulary_feature2


def test_feature2_performance():
    model_2, vocabulary_2 = adjust_feature2_n()
    accuracy, precision, recall, F_score = validation(model_2, vocabulary_2, X_test, Y_test)
    print("Accuracy: " + str(accuracy))
    print("macro-averaged precision: " + str(precision))
    print("macro-averaged recall: " + str(recall))
    print("macro-averaged F_score: " + str(F_score))


test_feature2_performance()