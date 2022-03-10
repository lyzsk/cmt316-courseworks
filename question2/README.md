# CMT316 coursework1 part1 question2

- Develop Enviornment
    - Python 3.9.7

- "real-state" folder
    - Store the original data

- Package
    - classification
    - regression

- classification.classification.py
    - use numpy, sklearn
    - get dataset from "real-state" folder
    - select the 6 most relevant features as the data points
    - house price large or equal to 30 refers to exensive, otherwise not-expensive
    - running result:
        - precision: 0.835
        - accuracy: 0.867

- regression.regression.py
    - use pandas, sklearn, math
    - get dataset from "real-state" folder
    - use linearRegression model from sklearn
    - get the rmse value of the model
    - running result:
        - 8.501078429154513