import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm  # for creating residual qqplots

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from collections import Counter

# ========================================================================================== #
# Classification FUNCTIONS
# ========================================================================================== #

def build_classification_model(X, y, y_cut_threshold=0.666, model_name="RF"):
    
    if model_name == 'RF':
        clf = RandomForestClassifier()
        clf_param_grid = {"max_depth": range(1,4), "max_features":[1,2,'sqrt'], "n_estimators":[10,50,100,150]}
    else:
        # TO DO: throw exception
        return


    ## Binarize y into 2 bins

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


    ## Tune hyperparameters via grid search

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

    ## Build and evaluate best model

    evaluate_classifier_(X_train, X_test, y_bin_train, y_bin_test, gs.best_estimator_)


    ## Learning curve of final model

    estimate_learning_curve_(X, y_bin, gs.best_estimator_)

    ## Feature importances

    estimate_feature_importances_(X, y_bin, gs.best_estimator_)



def evaluate_classifier_(X_train, X_test, y_train, y_test, clf):
    
    print("Performance of model with best CV hyperparameter values:")
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


# ========================================================================================== #
# Regression modeling FUNCTIONS
# ========================================================================================== #


#scoring_fn, scoring_fn_name = 'r2', 'R2'
#scoring_fn, scoring_fn_name = 'neg_mean_absolute_error', 'MAE'

def build_regression_model(Xtrain, ytrain, Xtest, ytest, estimator, param_dict, 
                           cv=10, scoring_fn='r2', scoring_fn_name='R2',
                          show_output=False, show_plots=False):

    ## Tune hyperparameters via grid search

    '''
    if len(param_dict) == 1:
        res_ = []
        param_name = list(param_dict.keys())[0]
        param_vals = list(param_dict.values())[0]
        for param_val in param_vals:
            ans1 = cross_val_score(estimator.set_params(**{param_name:param_val}), Xtrain, ytrain, cv=cv, scoring=scoring_fn).mean()
            res_.append( (param_val, ans1) )

        df = pd.DataFrame(res_, columns=[param_name, scoring_fn_name])
        best_param_val = df[param_name][df[scoring_fn_name].argmax()]

        assert df[scoring_fn_name].max() == max([x[1] for x in res_])
        
        if show_output:
            print(f'Best value of {param_name}= {best_param_val}, with {scoring_fn_name}= {df[scoring_fn_name].max()}')
            print()

        if show_plots:
            df.plot.scatter(x=param_name, y=scoring_fn_name, c='b')
            plt.title(f'Best value of {param_name}: {best_param_val}')
            plt.xlabel(f'Hyperparameter {param_name}')
            plt.ylabel(scoring_fn_name)
            plt.show()

        # fit using best hyperparameter
        estimator.set_params(**{param_name:best_param_val})
        estimator.fit(Xtrain, ytrain)
    '''

    if len(param_dict) == 0:
        estimator.fit(Xtrain, ytrain)
    
    else:
        gs = GridSearchCV(estimator=estimator,
                          param_grid=param_dict,
                          cv=cv,
                          n_jobs=2,
                          scoring=scoring_fn
                         )

        gs.fit(Xtrain, ytrain)

        print(f'Best hyperparameter values: {gs.best_params_}, with average CV {scoring_fn_name}= {gs.best_score_}')

        # fit using best hyperparameter
        estimator.set_params(**gs.best_params_)
        estimator.fit(Xtrain, ytrain)


    ## Evaluate performance of best model
    CV_r2_avg, CV_r2_std, CV_mae_avg, CV_mae_std, train_r2, train_mae, test_r2, test_mae = evaluate_regression_model_(Xtrain, ytrain, Xtest, ytest, estimator, show_output=show_output)

    ## Learning curve of best model
    if show_plots:
        X = pd.concat([Xtrain, Xtest], axis=0)
        y = pd.concat([ytrain, ytest], axis=0)
        display_learning_curve(estimator, X, y, cv=cv, scoring_fn=scoring_fn, scoring_fn_name=scoring_fn_name)

    #### TO DO: put these in a named tuple instead
    return CV_r2_avg, CV_r2_std, CV_mae_avg, CV_mae_std, train_r2, train_mae, test_r2, test_mae, estimator


def evaluate_regression_model_(Xtrain, ytrain, Xtest, ytest, estimator, show_output=False, show_plots=False):

    cv1 = cross_val_score(estimator, Xtrain, ytrain, cv=5, scoring='r2')
    cv2 = cross_val_score(estimator, Xtrain, ytrain, cv=5, scoring='neg_mean_absolute_error')
    #cv3 = -cross_val_score(estimator, Xtrain, ytrain, cv=5, scoring='neg_mean_squared_error')

    CV_r2_avg, CV_r2_std = cv1.mean(), cv1.std()
    CV_mae_avg, CV_mae_std = -cv2.mean(), cv2.std()

    def helper_func(X,y):
        y_pred = estimator.predict(X)
        return (r2_score(y, y_pred),  ## == estimator.score(X, y)
                mean_absolute_error(y, y_pred))
    
    train_r2, train_mae = helper_func(Xtrain,ytrain)
    test_r2, test_mae = helper_func(Xtest,ytest)

    if show_output:
        print("\nPerformance of the selected model:")
        print()

        print("\tCross-validation performance:")   
        print(f"\tR2: {CV_r2_avg} +- {CV_r2_std}")
        print(f"\tMAE: {CV_mae_avg} +- {CV_mae_std}")
        #print(f"\RMSE: {CV_rmse_avg} +- {CV_rmse_std}")
        print()

        print('\tTrain set performance:')
        print("\tR2:", train_r2)
        print("\tMAE:", train_mae)
        #print("\tRMSE:", train_rmse)
        print()

        print('\tTest set performance:')
        print("\tR2:", test_r2)
        print("\tMAE:", test_mae)
        #print("\tRMSE:", test_rmse)
        print()

    if show_plots:
        plt.subplot(1,2,1)
        plt.scatter(y=ytrain, x=ytrain_pred, c='b')
        plt.plot([ytrain_pred.min(),ytrain_pred.max()], [ytrain_pred.min(),ytrain_pred.max()], 'y--', lw=2)
        plt.xlabel('Fitted')
        plt.ylabel('True')
        plt.title(f'Train set')
        plt.axis('equal')

        plt.subplot(1,2,2)
        plt.scatter(y=ytest, x=ytest_pred, c='g')
        plt.plot([ytest_pred.min(),ytest_pred.max()], [ytest_pred.min(),ytest_pred.max()], 'y--', lw=2)
        plt.xlabel('Fitted')
        plt.ylabel('True')
        plt.title(f'Test set')
        plt.axis('equal')

        plt.tight_layout()
        plt.show()

        plt.subplot(1,2,1)
        plt.scatter(x=ytrain_pred, y=ytrain-ytrain_pred, c='b')
        plt.plot([ytrain_pred.min(),ytrain_pred.max()], [0,0], 'y--', lw=2)
        plt.xlabel('Fitted')
        plt.ylabel('Residuals')
        plt.title(f'Train set')
        plt.axis('equal')

        plt.subplot(1,2,2)
        plt.scatter(x=ytest_pred, y=ytest-ytest_pred, c='g')
        plt.plot([ytest_pred.min(),ytest_pred.max()], [0,0], 'y--', lw=2)
        plt.xlabel('Fitted')
        plt.ylabel('Residuals')
        plt.title(f'Test set')
        plt.axis('equal')

        plt.tight_layout()
        plt.show()
    
        # Create QQ plot
        sm.qqplot(ytest-ytest_pred, line='45')
        plt.title(f'Test set; QQ plot of residuals')
        plt.show()

    return CV_r2_avg, CV_r2_std, CV_mae_avg, CV_mae_std, train_r2, train_mae, test_r2, test_mae

def display_learning_curve(estimator, X, y, cv, scoring_fn, scoring_fn_name):
  
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring_fn)

    learning_curve_df = pd.DataFrame(zip(train_scores.mean(axis=1),valid_scores.mean(axis=1)), 
                                     index=train_sizes, columns=[f'{scoring_fn_name}_train_scores',f'{scoring_fn_name}_val_scores'])

    learning_curve_df.plot()
    plt.title('Learning curve ' + estimator.__class__.__name__)
    plt.xlabel('train size')
    plt.ylabel(scoring_fn_name)
    plt.show()

