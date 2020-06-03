import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ========================================================================================== #
# PREDICTIVE MODELING FUNCTIONS
# ========================================================================================== #

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from collections import Counter


def build_classification_model(X, y, y_cut_threshold=0.666, model_name="RF"):
    
    if model_name == 'RF':
        clf = RandomForestClassifier()
        clf_param_grid = {"max_depth": range(1,4), "max_features":[1,2,'sqrt'], "n_estimators":[10,50,100,150]}
    else:
        # TO DO: throw exception
        return


    ## Binarize y into 2 bins
    #print("Binarization of target value ...")
    #print()

    cut_labels = ['low','high']
    q = y.quantile(y_cut_threshold)
    cut_bins = [y.min()-1, q , y.max()+1]
    y_bin = pd.cut(y, bins=cut_bins, labels=cut_labels)
    
    #print("Distribution of target before binarization:")
    #print(pd.Series(y).describe())
    #print()

    #print("Distribution of target after binarization:")
    #print(Counter(y_bin))
    #print()

    ## Split dataset into train and test sets
    X_train, X_test, y_bin_train, y_bin_test = train_test_split(X, y_bin, test_size=0.2, stratify=y_bin, random_state=404)


    ## Tune classifier via grid search

    #print("Grid search ...")
    #print()

    scoring = "roc_auc"  ## 'f1'

    gs = GridSearchCV(estimator=clf,
                      param_grid=clf_param_grid,
                      cv=5,
                      n_jobs=2,
                      scoring=scoring
                     )

    gs.fit(X_train, y_bin_train=="high" if scoring == 'f1' else y_bin_train)
    
    print("Best hyperparameter values: ")
    print(gs.best_params_)
    print()

    ## build and evaluate best model
    evaluate_classifier_(X_train, X_test, y_bin_train, y_bin_test, gs.best_estimator_)

    ## Learning curve of final model
    estimate_learning_curve_(X, y_bin, gs.best_estimator_)

    ## Feature importances
    estimate_feature_importances_(X, y_bin, gs.best_estimator_)


def evaluate_classifier_(X_train, X_test, y_train, y_test, clf):
    
    print("Performance of best classification model:")
    print()
    print("Cross-validation performance:")   
    print("CV mean F1:", cross_val_score(clf, X_train, y_train=='high', cv=5, scoring='f1').mean())
    print("CV mean ROC AUC:", cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean())
    print("CV mean Accuracy:", cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean())
    print()

    clf = clf.set_params(random_state=1)
    #print(clf)

    clf.fit(X_train, y_train)

    print("Test performance:")
    print("Accuracy:", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print("AUC:", roc_auc_score(y_test=="high", y_pred=="high"))
    print("Precision/Recall (classification report):")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(pd.crosstab(y_test, y_pred))
    print()

    ## visualize/analyze classification errors
    idx = y_test != y_pred
    print("Analyzing classification errors:")
    print("\tNumber of errors",idx.sum())
    print()
    df = pd.Series(clf.predict_proba(X_test)[:,0], index=y_test.index)
    df = pd.concat([y_test.loc[idx], df.loc[idx]], axis=1)
    df.columns = ['y_true', 'high_pred_prob']
    print(df.sort_values(by='high_pred_prob'))
    print()
    plt.show()


def estimate_learning_curve_(X, y, clf):

    #print("Learning curve of best classification model ...")
    #print()

    #print(clf)
    #print()

    train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, cv=10, scoring='roc_auc')
    #train_sizes, train_scores, valid_scores = learning_curve(clf, X, y=='high', cv=10, scoring='f1')

    learning_curve_df = pd.DataFrame(zip(train_scores.mean(axis=1),valid_scores.mean(axis=1)), 
                                     index=train_sizes, columns=['train_scores','val_scores'])
    learning_curve_df.plot()
    plt.xlabel('train size')
    plt.ylabel('ROC_AUC')
    plt.title('Learning Curve')
    plt.show()


def estimate_feature_importances_(X, y, clf):

    #print("Estimating feature importance base on best classification model ...")
    #print()

    n_samples = 30
    feature_importances = np.empty((n_samples, X.shape[1]))

    # change rand_state
    clf = clf.set_params(random_state=None)
    #print(clf)

    for i in range(n_samples):
        # re-fit
        clf.fit(X, y)
        feature_importances[i,:] = clf.feature_importances_

    mu = feature_importances.mean(axis=0)
    sigma = feature_importances.std(axis=0)
    df = pd.DataFrame(dict(errorbar1=mu-sigma, avg=mu, errorbar2=mu+sigma), index=X.columns)
    df.plot(rot=30)
    plt.title('Distribution of feature importances')
    plt.show()

    assert df.shape[0] == X.shape[1]

    df.avg.plot.barh()
    plt.title('Average Importance Coefficients')
    plt.show()

    u, v = X.mean(), df.avg
    plt.scatter(u, v)
    plt.xlabel('Average feature value')
    plt.ylabel('RF feature importance')
    plt.title('Correlation between features\' average value and importance')
    for i,x in enumerate(X.columns):
        plt.text(u[i], v[i], x, fontsize=8, color='green')
    plt.show()


def tune_regression_model_(X, y, reg):
    pass


def evaluate_regression_model_(X_train, X_test, y_train, y_test, reg):
    pass


