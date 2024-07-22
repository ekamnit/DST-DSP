# DST-DSP
This is the page for Data and the code
# Description of the Code
The code in the path ./DST-DSP/Programs/ is a comprehensive implementation of a self-training semi-supervised learning approach using decision tree classifiers to detect the severity of software defects. The program leverages the imbalanced-learn library for handling class imbalance and uses the pandas and NumPy libraries for data manipulation. Below is a detailed description of the major components and steps in the code:

#### 1. **Importing Necessary Libraries**:
- Imports necessary libraries such as pandas, numpy, sklearn for machine learning tasks, and imblearn for handling imbalanced datasets.
- Uses Google Colab's drive module to mount Google Drive and access the data stored there.

#### 2. **Data Loading**:
- Loads the dataset from an Excel file located in Google Drive.
- Filters the dataset to separate labeled and unlabeled data based on certain conditions. The unlabeled data are those instances where specific bug counts are zero.

#### 3. **Data Preparation**:
- Separates the features (X) and labels (y) from the labeled dataset.
- Handles class imbalance in the labeled dataset using the ADASYN technique to generate synthetic samples.

#### 4. **Self-Training Semi-Supervised Learning Approach**:
- **Initialization**: Sets up containers to store f1 scores and counts of pseudo-labeled data. Initiates a decision tree classifier with a grid search for hyperparameter tuning.
- **Iteration Loop**: The loop continues until there are no high-probability predictions left in the unlabeled data.
  - **Model Training**: Trains the decision tree classifier with the labeled data and evaluates its performance.
  - **Predicting Unlabeled Data**: Uses the trained classifier to predict labels and probabilities for the unlabeled data. High-confidence predictions (with probability > 0.99) are added to the training data as pseudo-labeled instances.
  - **Update Training Set**: Adds the pseudo-labeled data to the training set and removes them from the unlabeled set.

#### 5. **Performance Measures**:
- **Function Definition**: Defines a function `Function(clf)` to evaluate the trained classifier on a test dataset.
  - **Model Training**: Re-trains the classifier with the combined labeled and pseudo-labeled data.
  - **Model Evaluation**: Predicts labels for the test set and calculates various performance metrics, including f1 score, confusion matrix, and risk factors for different bug categories.
  - **Risk Factors**: Computes the risk factors for different bug categories based on the confusion matrix.
  - **Budget and Service Time**: Calculates metrics related to the saved budget and remaining service time, demonstrating the efficiency of the model in reducing maintenance costs.

### Grid Search for Hyperparameter Tuning

Grid search is employed to find the best hyperparameters for the decision tree classifier. The grid search involves the following steps:

1. **Define Parameter Grid**: Specifies the range of hyperparameters to search over.
2. **Initialize GridSearchCV**: Uses the `GridSearchCV` class from `scikit-learn` with the decision tree classifier and the defined parameter grid.
3. **Cross-Validation**: Performs cross-validation to evaluate the performance of each combination of hyperparameters.
4. **Fit and Evaluate**: Fits the grid search object to the training data and retrieves the best hyperparameters based on cross-validation scores.

### Code Walkthrough

#### Import Libraries and Load Data
```python
import os
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_val_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_excel('/content/gdrive/MyDrive/Work2-CPDP-Replication/AEEEM/eclipse.xlsx')
```

#### Data Preparation
```python
unlabelled_df = df.loc[(df['Nondefective'] == 0) & (df[' highPriorityBugs '] == 0) & (df[' criticalBugs '] == 0) & (df[' majorBugs '] == 0) & (df[' nonTrivialBugs '] == 0)]
labelled_df = df.loc[~((df['Nondefective'] == 0) & (df[' highPriorityBugs '] == 0) & (df[' criticalBugs '] == 0) & (df[' majorBugs '] == 0) & (df[' nonTrivialBugs '] == 0))]
```

#### Handling Imbalanced Data
```python
oversample = ADASYN(n_neighbors=1)
X, Y = oversample.fit_resample(x, y)
```

#### Self-Training Loop
```python
iterations = 0
train_f1s = []
test_f1s = []
pseudo_labels = []
high_prob = [1]
Y = np.argmax(Y, axis=1)

while len(X_unlabelled) > 0:
    clf = DecisionTreeClassifier()
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, Y)
    best_params = grid_search.best_params_
    clf.set_params(**best_params)
    clf.fit(X, Y)
    
    y_hat_train = clf.predict(X)
    train_f1 = f1_score(Y, y_hat_train, average='micro')
    train_f1s.append(train_f1)

    preds = clf.predict(X_unlabelled)
    pred_probs = clf.predict_proba(X_unlabelled)

    df_pred_prob = pd.DataFrame(pred_probs, columns=[f'prob_{i}' for i in range(pred_probs.shape[1])])
    df_pred_prob['preds'] = preds

    high_prob = df_pred_prob[(df_pred_prob > 0.99).any(axis=1)]
    pseudo_labels.append(len(high_prob))

    if not high_prob.empty:
        X = np.vstack((X, X_unlabelled[high_prob.index]))
        Y = np.hstack((Y, high_prob['preds'].values))

        X_unlabelled = np.delete(X_unlabelled, high_prob.index, axis=0)

    iterations += 1
```

#### Performance Measures
```python
def Function(clf):
    clf.fit(X, Y)
    y_hat_train = clf.predict(X)
    train_f1 = f1_score(Y, y_hat_train, average='micro')
    print(f"Train f1: {train_f1}")

    df_test = pd.read_excel('/content/gdrive/MyDrive/Btech_Project/CombinedDataset/pde.xlsx')
    labelled_df = df_test.loc[~((df_test['Nondefective'] == 0) & (df_test[' highPriorityBugs '] == 0) & (df_test[' criticalBugs '] == 0) & (df_test[' majorBugs '] == 0) & (df_test[' nonTrivialBugs '] == 0))]
    data = labelled_df.values
    x_test = data[:, idx_IN_columns]
    x_test = x_test[:, 1:]
    y_test = data[:, idx_OUT_columns]
    y_test = np.argmax(y_test.astype('int'), axis=1)

    yhat_test = clf.predict(x_test)
    yhat_pred_probs = clf.predict_proba(x_test)
    test_f1 = f1_score(y_test, yhat_test, average='micro')
    print(f"Test f1: {test_f1}")

    CM = confusion_matrix(y_test, yhat_test, labels=[0, 1, 2, 3, 4])
    Risk_HP = ((0.1 * CM[0][1]) + (0.2 * CM[0][2]) + (0.3 * CM[0][3]) + (0.4 * CM[0][4])) / CM[0].sum()
    Risk_C = ((0.1 * CM[1][2]) + (0.2 * CM[1][3]) + (0.3 * CM[1][4])) / CM[1].sum()
    Risk_M = ((0.1 * CM[2][3]) + (0.2 * CM[2][4])) / CM[2].sum()
    Risk_NT = (0.1 * CM[3][4]) / CM[3].sum()

    print(f"Risk Factor for High Priority: {Risk_HP}")
    print(f"Risk Factor for Critical: {Risk_C}")
    print(f"Risk Factor for Major: {Risk_M}")
    print(f"Risk Factor for Non-Trivial: {Risk_NT}")

    total_loc = x_test[:, 28].sum()
    tn_loc = x_test[(yhat_test == 4) & (y_test == 4), 28].sum()
    print(f"Saved Budget: {tn_loc}")
    print(f"PSB: {tn_loc / total_loc}")
    print(f"PNTN:

 {1 - (CM[4][4] / len(x_test))}")
    print(f"Remaining Service Time: {total_loc - tn_loc}")
    print(f"PRST: {1 - (tn_loc / total_loc)}")

Function(LogisticRegression())
Function(DecisionTreeClassifier())
```

This comprehensive approach combines advanced machine learning techniques with practical performance evaluation to create a robust model for defect prediction in software projects.
