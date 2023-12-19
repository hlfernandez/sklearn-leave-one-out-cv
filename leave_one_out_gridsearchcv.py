from sklearn.datasets import  make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def leave_one_out_grid_search_cv(X, y, model, param_grid):
    """
    Perform Leave-One-Out cross-validated grid search for hyperparameter tuning.

    :param X: The feature matrix.
    :type X: array-like or pd.DataFrame

    :param y: The target variable.
    :type y: array-like or pd.Series

    :param model: The machine learning model for which hyperparameter tuning is performed.
    :type model: Any

    :param param_grid: The parameter grid to search over.
    :type param_grid: dict

    :return: A scikit-learn GridSearchCV object configured for Leave-One-Out
        cross-validation and hyperparameter tuning.
    :rtype: sklearn.model_selection.GridSearchCV
    """
    def get_pred(y_true, y_predicted):
        return y_predicted

    pred_scorer = make_scorer(get_pred)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, refit=False, scoring=pred_scorer, cv=LeaveOneOut())
    grid_search.fit(X, y)

    return grid_search

def print_params_scores(grid_search, y_true, scorer_fun, scorer_fun_args = {}):
    """
    Print the scores of each parameter set obtained during cross-validation.

    :param grid_search: The scikit-learn GridSearchCV object.
    :type grid_search: sklearn.model_selection.GridSearchCV

    :param y_true: The true target variable values.
    :type y_true: array-like

    :param scorer_fun: The scoring function to evaluate predictions. It should take
        `y_true` and `y_pred` as input.
    :type scorer_fun: callable

    :param scorer_fun_args: Additional arguments to be passed to the `scorer_fun`.
    :type scorer_fun_args: dict, optional
    :default scorer_fun_args: {}

    :returns: None
    :rtype: None
    """
    y_preds = []
    results = grid_search.cv_results_
    params = results['params']

    for j in range(len(params)):
        y_preds.append([])
        for i in range(grid_search.n_splits_):
            prediction_of_j_on_i = results[f'split{i}_test_score'][j]
            y_preds[j].append(prediction_of_j_on_i)

    for j in range(len(y_preds)):
        score = scorer_fun(y_true, y_preds[j], **scorer_fun_args)
        print(f'{params[j]} obtained {scorer_fun.__name__} of {score}')

def find_best_params(grid_search, y_true, scorer_fun, scorer_fun_args = {}):
    """
    Find the best parameters based on cross-validated scores.

    :param grid_search: The scikit-learn GridSearchCV object.
    :type grid_search: sklearn.model_selection.GridSearchCV

    :param y_true: The true target variable values.
    :type y_true: array-like

    :param scorer_fun: The scoring function to evaluate predictions. It should take
        `y_true` and `y_pred` as input.
    :type scorer_fun: callable

    :param scorer_fun_args: Additional arguments to be passed to the `scorer_fun`.
    :type scorer_fun_args: dict, optional
    :default scorer_fun_args: {}

    :returns: A tuple containing the best score and corresponding best parameters.
    :rtype: tuple
    """
    y_preds = []
    results = grid_search.cv_results_
    params = results['params']

    for j in range(len(params)):
        y_preds.append([])
        for i in range(grid_search.n_splits_):
            prediction_of_j_on_i = results[f'split{i}_test_score'][j]
            y_preds[j].append(prediction_of_j_on_i)

    best_score = None
    for j in range(len(y_preds)):
        score = scorer_fun(y_true, y_preds[j], **scorer_fun_args)
        if best_score is None or best_score < score:
            best_score = score
            best_params = params[j]

    return best_score, best_params

def merge_params_dict(best_params_dict, model_params_dict):
    """
    Merge two dictionaries of parameters.

    :param best_params_dict: The dictionary containing the best parameters.
    :type best_params_dict: dict

    :param model_params_dict: The dictionary containing additional parameters.
    :type model_params_dict: dict

    :returns: A new dictionary containing a merged set of parameters. If a parameter
        exists in best_params_dict, it is retained; otherwise, it is taken from
        model_params_dict.
    :rtype: dict
    """
    toret = best_params_dict.copy()
    for key in model_params_dict.keys():
        if not toret.get(key):
            toret[key] = model_params_dict[key]
    
    return toret

def recreate_model_with_best_params(original_model, best_params_dict):
    """
    Recreate a model instance with the best parameters.

    :param original_model: The original model instance to be recreated. It is expected to have a 'get_params' method.
    :type original_model: Any

    :param best_params_dict: The dictionary containing the best parameters.
    :type best_params_dict: dict

    :returns: A new instance of the original model with parameters updated based on the best_params_dict.
    :rtype: Any
    """
    return type(original_model)(**merge_params_dict(best_params_dict, original_model.get_params()))

def recreate_pipeline_with_best_params(original_pipeline, best_params_dict, verbose=False):
    """
    Recreate a pipeline with the best parameters for each step.

    :param original_pipeline: The original scikit-learn pipeline to be recreated.
    :type original_pipeline: sklearn.pipeline.Pipeline

    :param best_params_dict: The dictionary containing the best parameters.
    :type best_params_dict: dict

    :param verbose: If True, print information about each recreated step. Default is False.
    :type verbose: bool, optional

    :return: A new instance of the pipeline with parameters updated based on the
        best_params_dict for each step.
    :rtype: sklearn.pipeline.Pipeline
    """
    best_steps = []
    for step in original_pipeline.steps:
        filtered_dict = {key.replace(f'{step[0]}__', ''): value for key, value in best_params_dict.items() if key.startswith(f'{step[0]}__')}

        step_model = recreate_model_with_best_params(step[1], filtered_dict)
        best_steps.append((step[0],step_model))

        if verbose:
            print('*' * 10)
            print('Pipeline step: ', step[0])
            print('Parameters: ', filtered_dict)
            print('Instance: ', step_model)
    
    return Pipeline(steps=best_steps)


if __name__ == '__main__':
    random_state=2023

    # Example 1: Binary classification with a decision tree

    model = DecisionTreeClassifier(random_state=random_state, max_depth=5)
    param_grid = dict(criterion=['gini', 'entropy'], min_samples_leaf=[6, 12, 18])

    X, y = make_classification(n_samples=100, random_state=1, n_classes = 2, n_informative=3, n_features=12)
    loo_grid_search_cv = leave_one_out_grid_search_cv(X, y, model, param_grid)

    model_name = type(model).__name__
    print('\n*** Binary classification ***')
    print(f'\n{model_name} F1-scores:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = f1_score)


    print(f'\n{model_name} accuracies:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = accuracy_score)

    best_score, best_params = find_best_params(loo_grid_search_cv, y, scorer_fun = f1_score)
    print(f'\n{model_name} best F1-score was {best_score} with the following parameters: {best_params}')

    new_best_model = recreate_model_with_best_params(model, best_params)
    print(new_best_model)
    
    # Example 2: Three-class classification with decision tree
    
    model = DecisionTreeClassifier(random_state=random_state, max_depth=5)
    param_grid = dict(criterion=['gini', 'entropy'], min_samples_leaf=[6, 12, 18])

    X, y = make_classification(n_samples=100, random_state=1, n_classes = 3, n_informative=3, n_features=12)
    loo_grid_search_cv = leave_one_out_grid_search_cv(X, y, model, param_grid)

    model_name = type(model).__name__
    print('\n*** Three-class classification ***')
    print(f'\n{model_name} F1-scores:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = f1_score, scorer_fun_args = {'average' : 'macro'})

    print(f'\n{model_name} accuracies:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = accuracy_score)

    best_score, best_params = find_best_params(loo_grid_search_cv, y, scorer_fun = f1_score, scorer_fun_args = {'average' : 'macro'})
    print(f'{model_name} best F1-score was {best_score} with the following parameters: {best_params}')

    new_best_model = recreate_model_with_best_params(model, best_params)
    print(new_best_model)

    # Example 3: Binary classification using a pipeline

    X, y = make_classification(n_samples=100, random_state=1, n_classes = 2, n_informative=3, n_features=12)

    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    scaler = StandardScaler(with_std=False)
    
    param_grid = {
        "logistic__C": np.logspace(-4, 4, 4),
        "scaler__with_mean": [True, False]
    }

    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

    loo_grid_search_cv = leave_one_out_grid_search_cv(X, y, pipe, param_grid)

    best_score, best_params = find_best_params(loo_grid_search_cv, y, scorer_fun = f1_score)
    
    print(f'Pipeline best F1-score was {best_score} with the following parameters: {best_params}')

    best_pipeline = recreate_pipeline_with_best_params(pipe, best_params, verbose=True)
    print(best_pipeline)
