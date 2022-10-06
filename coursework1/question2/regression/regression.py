import pandas
import sklearn
from sklearn.linear_model import LinearRegression
import math

# get train data
X_train = pandas.read_csv("../real-state/train_full_Real-estate.csv", index_col=0).iloc[:, :-1]
Y_train = pandas.read_csv("../real-state/train_full_Real-estate.csv", index_col=0).iloc[:, -1]

# train model
svm_clf_house = LinearRegression()
svm_clf_house.fit(X_train.to_numpy(), Y_train.to_numpy())

# get test data
X_test = pandas.read_csv("../real-state/test_full_Real-estate.csv", index_col=0).iloc[:, :-1]
Y_test = pandas.read_csv("../real-state/test_full_Real-estate.csv", index_col=0).iloc[:, -1]

Y_predictions = svm_clf_house.predict(X_test.to_numpy())
Y_test_numpy = Y_test.to_numpy()
# rmse = ((Y_predictions - Y_test_numpy)**2).mean()**0.5
# print(rmse)
print(math.sqrt(sklearn.metrics.mean_squared_error(Y_test_numpy, Y_predictions)))