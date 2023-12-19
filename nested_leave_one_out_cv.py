
from sklearn.model_selection import LeaveOneOut
from leave_one_out_gridsearchcv import leave_one_out_grid_search_cv, find_best_params, recreate_pipeline_with_best_params, recreate_model_with_best_params
import numpy as np

def get_current_split(X, y, train_ix, test_ix):
    """
    Get the current split of data for training and testing.

    :param X: The feature matrix.
    :type X: array-like or pd.DataFrame

    :param y: The target variable.
    :type y: array-like or pd.Series

    :param train_ix: Indices of the training data.
    :type train_ix: array-like

    :param test_ix: Indices of the testing data.
    :type test_ix: array-like

    :returns: A tuple containing the current split of data for training and testing.
        If X and y are DataFrames, the resulting splits will also be DataFrames;
        otherwise, they will be arrays.
    :rtype: tuple
    """
    if type(X).__name__ == 'DataFrame':
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    else:
        X_train, X_test = X[train_ix, :], X[test_ix, :]
    
    if type(y).__name__ == 'DataFrame':
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    else:
        y_train, y_test = y[train_ix], y[test_ix]
    
    return X_train, X_test, y_train, y_test

def nested_loocv(
    model, 
    X, 
    y, 
    param_grid, 
    scorer_fun, 
    scorer_fun_args={},
    verbose=False
):
    """
    Perform nested Leave-One-Out cross-validation for hyperparameter tuning.

    :param model: The machine learning model for which hyperparameter tuning is performed.
    :type model: Any

    :param X: The feature matrix.
    :type X: array-like or pd.DataFrame

    :param y: The target variable.
    :type y: array-like or pd.Series

    :param param_grid: The parameter grid to search over.
    :type param_grid: dict

    :param scorer_fun: The scoring function to evaluate predictions. It should take
        `y_true` and `y_pred` as input.
    :type scorer_fun: callable

    :param scorer_fun_args: Additional arguments to be passed to the `scorer_fun`.
    :type scorer_fun_args: dict, optional
    :default scorer_fun_args: {}

    :param verbose: If True, print information about each iteration. Default is False.
    :type verbose: bool, optional

    :returns: A tuple containing the true target values (`y_true`) and predicted
        target values (`y_pred`) for each iteration of the nested cross-validation.
    :rtype: tuple
    """
    cv = LeaveOneOut()
    y_true, y_pred = list(), list()

    for train_ix, test_ix in cv.split(X): # Outer loop
        X_train, X_test, y_train, y_test = get_current_split(X, y, train_ix, test_ix)

        # Inner loop is done via CV using X_train and y_train
        loo_grid_search_cv = leave_one_out_grid_search_cv(X_train, y_train, model, param_grid)

        best_score, best_params = find_best_params(loo_grid_search_cv, y_train, scorer_fun=scorer_fun, scorer_fun_args=scorer_fun_args)

        if type(model).__name__ == 'Pipeline':
            new_best_model = recreate_pipeline_with_best_params(model, best_params)
        else:
            new_best_model = recreate_model_with_best_params(model, best_params)

        if verbose:
            print(best_score, new_best_model)

        # Train it with the whole train partition and predict the test sample
        new_best_model.fit(X_train, y_train)
        yhat = new_best_model.predict(X_test)
        y_true.append(y_test[0])
        y_pred.append(yhat[0])

    return y_true, y_pred
