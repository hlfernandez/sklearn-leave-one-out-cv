import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, classification_report
from nested_leave_one_out_cv import nested_loocv

if __name__ == '__main__':
    random_state = 2023

    # Binary classification
    X, y = make_classification(n_samples=50, random_state=random_state, n_classes = 2, n_informative=3, n_features=12)
    target_names = list(map(str, np.unique(y)))

    # Example 1: a single decision tree

    model = DecisionTreeClassifier(random_state=random_state)
    param_grid = dict(criterion=['gini', 'entropy'], min_samples_leaf=[6, 12, 18])

    y_true, y_pred = nested_loocv(model, X, y, param_grid, scorer_fun=f1_score)

    print('DecisionTreeClassifier classification report (outer CV):')    
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    # plt.show()

    # Example 2: a pipeline with feature selection and a decision tree

    dt = DecisionTreeClassifier(random_state=random_state, max_depth=5)    
    fs = SelectKBest(f_classif)
    pipeline_1 = Pipeline(steps=[("fs", fs), ("dt", dt)])

    param_grid = {
        "fs__k": [2, 4, 6, 12],
        "dt__criterion": ['gini', 'entropy'],
        "dt__min_samples_leaf": [6, 12, 18]
    }
    
    y_true, y_pred = nested_loocv(pipeline_1, X, y, param_grid, scorer_fun=f1_score)
        
    print('Pipeline (SelectKBest + DecisionTreeClassifier) classification report (outer CV):')
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Example 3: a pipeline with scaling and logistic regression
    
    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    scaler = StandardScaler()
    pipeline_2 = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

    param_grid = {
        "logistic__C": np.logspace(-4, 4, 4)
    }

    y_true, y_pred = nested_loocv(pipeline_2, X, y, param_grid, scorer_fun=f1_score)
        
    print('Pipeline (StandardScaler + LogisticRegression) classification report (outer CV):')
    print(classification_report(y_true, y_pred, target_names=target_names))
