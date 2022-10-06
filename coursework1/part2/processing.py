import os
import pandas
import sklearn.model_selection

# create a dictionary to store 5 categories
category_dic = {
    "business": 1,
    "entertainment": 2,
    "politics": 3,
    "sport": 4,
    "tech": 5
}


# read all txt files in the directory and use pandas.DataFrame() to categorize them
def read_files(path, category_id):
    result = []
    file_names = os.listdir(path)
    for file_name in file_names:
        if ".txt" in file_name:
            file = open(path + "/" + file_name)
            result.append([file.read(), category_id])
            file.close()
    return pandas.DataFrame(result, columns=["content", "category"])


raw_data = []
for category, category_id in category_dic.items():
    raw_data.append(read_files("bbc/" + category, category_id))
dataset = pandas.concat(raw_data, ignore_index=True)
X_dataset = dataset.iloc[:, :-1]
Y_dataset = dataset.iloc[:, -1]

# print(X_dataset.head(5))

# split train/test/dev data into 80%/10%/10%
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_dataset, Y_dataset, test_size=0.2, random_state=42)
X_dev, X_test, Y_dev, Y_test = sklearn.model_selection.train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

# print(X_train.head(5))
# print(Y_train.head(5))