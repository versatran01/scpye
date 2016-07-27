from __future__ import (print_function, absolute_import, division)

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier)
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from scpye.utils.exception import ClassifierNotSupportedError


def create_single_classifier(clf_name='svc'):
    """
    Creates a single classifier
    :param clf_name: short name of classifier
    """
    if clf_name == 'svc':
        clf = SVC()
        params = {'C': [20, 200]}
    elif clf_name == 'lr':
        clf = LogisticRegression()
        params = {'C': [20, 200]}
    elif clf_name == 'rf':
        clf = RandomForestClassifier()
        params = {'n_estimators': [20, 50]}
    else:
        raise ClassifierNotSupportedError(clf_name)

    return clf, params


def create_voting_classifier(voting='hard', classifiers=('svc', 'lr', 'rf')):
    """
    Creates a voting classifier
    :param voting: hard or soft, hard is much faster
    :param classifiers: tuple of classifiers
    :rtype: tuple(VotingClassifier, dict)
    """
    if voting == 'soft':
        raise NotImplementedError

    estimators = []
    param_grid = {}

    for clf_name in classifiers:
        clf, params = create_single_classifier(clf_name)
        estimators.append((clf_name, clf))
        # add params to param_grid
        for k, v in params.items():
            k = clf_name + '__' + k
            param_grid[k] = v

    eclf = VotingClassifier(estimators=estimators, voting=voting)
    return eclf, param_grid


def cross_validate_classifier(X, y, clf, param_grid, test_size=0.3,
                              report=True):
    """
    :rtype: GridSearchCV
    """
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=test_size)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=4, verbose=5,
                        scoring="f1_weighted")
    grid.fit(X_t, y_t)
    if report:
        print_grid_search_report(grid)
        print_validation_report(X_v, y_v, grid)

    return grid


def print_grid_search_report(grid):
    """
    Print grid search report
    :type grid: GridSearchCV
    """
    print("")
    print("Grid search cross validation report:")
    print("All parameters searched:")
    for params, mean_score, scores in grid.grid_scores_:
        print("{0:03f} (+/-{1:03f}) - {2}".format(mean_score, scores.std() * 2,
                                                  params))
    print("")
    print("Optimal parameters and best score:")
    print("{0:06f} for {1}".format(grid.best_score_, grid.best_params_))
    print("")


def print_validation_report(X, y, clf, target_names=None):
    """
    Print classification report
    :type X: np.ndarray
    :type y: np.ndarray
    :type clf: GridSearchCV
    :param target_names:
    """
    if target_names is None:
        target_names = ['Non-apple', 'Apple']
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred, target_names=target_names)
    print(report)
