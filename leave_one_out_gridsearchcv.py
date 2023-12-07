from sklearn.datasets import  make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

def leave_one_out_grid_search_cv(X, y, model, param_grid):
    def get_pred(y_true, y_predicted):
        return y_predicted

    pred_scorer = make_scorer(get_pred)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, refit=False, scoring=pred_scorer, cv=LeaveOneOut())
    grid_search.fit(X, y)

    return grid_search

def print_params_scores(grid_search, y_true, scorer_fun, scorer_fun_args = {}):
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

if __name__ == '__main__':
    random_state=2023
    model = DecisionTreeClassifier
    param_grid = dict(criterion=['gini', 'entropy'], min_samples_leaf=[6, 12, 18])

    # Binary classification
    X, y = make_classification(n_samples=100, random_state=1, n_classes = 2, n_informative=3, n_features=12)
    loo_grid_search_cv = leave_one_out_grid_search_cv(X, y, model(random_state=random_state), param_grid)

    print('\n*** Binary classification ***')
    print(f'\n{model.__name__} F1-scores:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = f1_score)

    print(f'\n{model.__name__} accuracies:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = accuracy_score)


    best_score, best_params = find_best_params(loo_grid_search_cv, y, scorer_fun = f1_score)
    print(f'\n{model.__name__} best F1-score was {best_score} with the following parameters: {best_params}')

    # Create a new classifier with the best params
    new_best_model = model(random_state=random_state, **best_params)
    print(new_best_model)

    # Three-class classification
    X, y = make_classification(n_samples=100, random_state=1, n_classes = 3, n_informative=3, n_features=12)
    loo_grid_search_cv = leave_one_out_grid_search_cv(X, y, model(random_state=random_state), param_grid)

    print('\n*** Three-class classification ***')
    print(f'\n{model.__name__} F1-scores:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = f1_score, scorer_fun_args = {'average' : 'macro'})

    print(f'\n{model.__name__} accuracies:')
    print_params_scores(loo_grid_search_cv, y, scorer_fun = accuracy_score)

    best_score, best_params = find_best_params(loo_grid_search_cv, y, scorer_fun = f1_score, scorer_fun_args = {'average' : 'macro'})
    print(f'{model.__name__} best F1-score was {best_score} with the following parameters: {best_params}')

    # Create a new classifier with the best params
    new_best_model = model(random_state=random_state, **best_params)
    print(new_best_model)