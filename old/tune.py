from sklearn import svm
from sklearn import linear_model
from sklearn import grid_search
from sklearn import ensemble


def tune_svc(X, y):
    """
    Tune support vector classifier
    :param X:
    :param y:
    :return:
    """
    params = [
        {'kernel': ['rbf'], 'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}]

    grid = grid_search.GridSearchCV(estimator=svm.SVC(), param_grid=params,
                                    cv=5)
    grid.fit(X, y)

    return grid


def tune_lr(X, y):
    """
    Tune logistic regression classifier
    :param X:
    :param y:
    :return:
    """
    params = {'C': [1, 5, 10, 50, 100]}
    grid = grid_search.GridSearchCV(estimator=linear_model.LogisticRegression(),
                                    param_grid=params, cv=5)
    grid.fit(X, y)

    return grid


def tune_rf(X, y):
    """
    Tune random forest classifier
    :param X:
    :param y:
    :return:
    """
    params = {'n_estimators': [5, 10, 25, 50, 100]}
    grid = grid_search.GridSearchCV(estimator=ensemble.RandomForestClassifier(),
                                    param_grid=params, cv=5)
    grid.fit(X, y)

    return grid


def tune_ensemble(X, y):
    """
    Tune ensemble classifier
    :param X:
    :param y:
    :return:
    """
    clf_svc = svm.SVC()
    clf_lr = linear_model.LogisticRegression()
    clf_rf = ensemble.RandomForestClassifier()

    eclf = ensemble.VotingClassifier(
        estimators=[('lr', clf_lr), ('svc', clf_svc), ('rf', clf_rf)],
        voting='hard')
    params = {'svc__kernel': ['rbf'], 'svc__C': [1, 5, 10, 50, 100],
              'lr__C': [1, 5, 10, 50, 100],
              'rf__n_estimators': [5, 10, 25]}

    grid = grid_search.GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(X, y)

    return grid


def print_grid_search_report(grid):
    """
    :param grid:
    :return:
    """
    print("All Parameters Searched:")
    for params, mean_score, scores in grid.grid_scores_:
        print(
            "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print(" ")

    print("Optimal Parameters:")
    print (grid.best_params_)

    print("Best Score:")
    print(grid.best_score_)
