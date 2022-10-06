import os
# import pandas
import numpy
import sklearn.svm
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report


# function to get file path
def get_path(string):
    # read file from parent directory
    path = os.path.join(os.path.abspath(".."), string)
    return path


# function to get dataset
def get_dataset(string):
    dataset = open(get_path(string)).readlines()
    return dataset


dataset_file = get_dataset("real-state/train_full_Real-estate.csv")

X_train = []
Y_train = []
# 0. No
# 1. X1 transaction date
# 2. X2 house age
# 3. X3 distance to the nearest MRT station
# 4. X4 number of convenience stores
# 5. X5 latitude
# 6. X6 longitude
# 7. Y house price of unit area
selected_features = [0, 1, 2, 3, 4, 5, 6]
# ignore the first row in dataset_file
for house_line in dataset_file[1:]:
    house_line_split = house_line.strip().split(',')
    vector_house_features = numpy.zeros(len(selected_features))
    feature_index = 0
    for i in range(len(house_line_split) - 1):
        if i in selected_features:
            vector_house_features[feature_index] = float(house_line_split[i])
            feature_index += 1
    X_train.append(vector_house_features)
    # expensive: 1 (higher or equal to 30)
    # not-expensive: 0 (lower than 30)
    if float(house_line_split[-1]) >= 30.0:
        Y_train.append(1)
    else:
        Y_train.append(0)

# convert the info list into numpy arrays
X_train_house = numpy.asarray(X_train)
Y_train_house = numpy.asarray(Y_train)

# keep only 6 most relevant features
feature_selection = SelectKBest(chi2, k=6).fit(X_train_house, Y_train_house)
X_train_house_new = feature_selection.transform(X_train_house)
svm_clf_house_new = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_house_new.fit(X_train_house_new, Y_train_house)


# print [1] if expensive, otherwise, print [0]
def test_clf(test_list):
    print(svm_clf_house_new.predict([test_list]))


# # 1
# test_clf(["2013.083", "42.7", "443.802", "6", "24.97927", "121.53874"])
# # 0
# test_clf(["2013.417", "31.5", "5512.038", "1", "24.95095", "121.48458"])

X_test = []
Y_test = []
test_file = get_dataset("real-state/test_full_Real-estate.csv")
for house_line in test_file[1:]:
    test_house_line_split = house_line.strip().split(',')
    test_vector_house_features = numpy.zeros(len(selected_features))
    test_feature_index = 0
    for i in range(len(test_house_line_split) - 1):
        if i in selected_features:
            test_vector_house_features[test_feature_index] = float(test_house_line_split[i])
            test_feature_index += 1
    X_test.append(test_vector_house_features)
    # expensive: 1 (higher or equal to 30)
    # not-expensive: 0 (lower than 30)
    if float(test_house_line_split[-1]) >= 30.0:
        Y_test.append(1)
    else:
        Y_test.append(0)

# convert the info list into numpy arrays
X_test_house = numpy.asarray(X_test)
Y_test_house = numpy.asarray(Y_test)

# keep only 6 most relevant features
feature_selection_test = SelectKBest(chi2, k=6).fit(X_test_house, Y_test_house)
X_test_house_new = feature_selection.transform(X_test_house)
svm_clf_test_new = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_test_new.fit(X_test_house_new, Y_test_house)

Y_test_predictions = svm_clf_test_new.predict(X_test_house_new)
print(classification_report(Y_test, Y_test_predictions))

from sklearn.metrics import precision_score, accuracy_score

precision = precision_score(Y_test, Y_test_predictions, average="macro")
accuracy = accuracy_score(Y_test, Y_test_predictions)
print("precision: " + str(round(precision, 3)))
print("accuracy: " + str(round(accuracy, 3)))
