# Classifiers
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM, LinearSVC, NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,\
                            GradientBoostingClassifier, BaggingClassifier

# Transformers
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer

# Decomposers
from sklearn.decomposition import PCA, KernelPCA, FastICA, SparsePCA, IncrementalPCA, TruncatedSVD, MiniBatchSparsePCA
from sklearn.cluster import FeatureAgglomeration

# Model selection, evaluation, and other stats functions
from sklearn.model_selection import train_test_split, cross_val_score,\
                                StratifiedShuffleSplit, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from scipy import stats
from scipy.stats import ttest_1samp
import statsmodels.api as sm


# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Others
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import time
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning, RuntimeWarning))


# Define functions
def load_data(file_name, folder_name=None):
    if folder_name != None:
        path = Path("/".join([os.getcwd(), folder_name, file_name]))
    else:
        path = Path("/".join([os.getcwd(), file_name]))
    return pd.read_csv(path)


def plot_histograms(data, target, target_figsize, dependent_layout, dependent_figsize):
    print(f"Distribution of target {target} and dependent variables:")
    data[target].hist(figsize=target_figsize, grid=False).set_title(target)
    # data.iloc[:, [1,2,3,4,5,6,0]].hist(layout=(2,3), figsize=(8,6), sharey=True, grid=False)
    data.drop([target], axis=1).hist(layout=dependent_layout, figsize=dependent_figsize, sharey=True, grid=False)
    plt.tight_layout();


def run_chi_tests(data, target, significance_level,
                  plot_row, plot_col, figsize, plot=True,
                  goodness_of_fit_test=True):
    chi_independence_df = pd.DataFrame(columns=[
        "Independent Variable",
        "Chi-square",
        "P-value",
        "Null Hypothesis",
        f"Reject Null Hypothesis at alpha={significance_level}?"
        ])
    
    if goodness_of_fit_test:
        print("----------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------")
        print("1. Chi-square test of goodness of fit")
        print("----------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------")

    if plot:
        fig, axes = plt.subplots(plot_row, plot_col, figsize=figsize, sharey=True)
        fig.tight_layout(h_pad=3)
        plt.suptitle(f"Chart 1. Relationship between {target} and other variables", y=1.05)
    for i, col in enumerate(data.drop(target, axis=1).columns):
        if plot:
            ax = axes[i//plot_col, i%plot_col]
            sns.lineplot(data, x=col, y=target, ax=ax).invert_yaxis()
            ax.set_yticks(sorted(list(data[target].unique())))
            ax.set_ylabel(target, rotation=0)
        x = data[col]
        y = data[target]

        if goodness_of_fit_test:
            contingency_table = pd.crosstab(x, y)
            print(f'Contingecy table for {col} and {target}:')
            print(contingency_table, "\n")

            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            print(f'Expected frequencies for {col} and {target}:')
            print(expected)

            "1. Perform Chi-square test of goodness of fit and print out the result."
            print(f"\nTesting goodness of fit for {col}.")
            chi_goodness_of_fit_test(x, col, significance_level)

            "2. Perform Chi-Square test of Independence and store the result in a dataframe."
            chi_independence_df = chi_independence_test(chi_independence_df, col, target, chi2, p, significance_level)
            print("--------------------------------------")
        else:
            contingency_table = pd.crosstab(x, y)
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            "1. Perform Chi-Square test of Independence and store the result in a dataframe."
            chi_independence_df = chi_independence_test(chi_independence_df, col, target, chi2, p, significance_level)

    return chi_independence_df


def chi_goodness_of_fit_test(data, col, significance_level):
    chi_goodness_of_fit_result = stats.chisquare(data)

    goodness_of_fit_null_hypothesis = f'There is no significant difference between {col} and the expected frequencies'
    if chi_goodness_of_fit_result.pvalue <= significance_level:
        goodness_of_fit_result = f""""
Null hypothesis: {goodness_of_fit_null_hypothesis}
Chi-square statistic: {chi_goodness_of_fit_result.statistic}
P-value: {chi_goodness_of_fit_result.pvalue}
Reject the null hypothesis
=> {col} is not representative of the population at alpha={significance_level}."""
    
    else: # Fail to reject the null hypothesis
        goodness_of_fit_result = f"""
Null hypothesis: {goodness_of_fit_null_hypothesis}
Chi-square statistic: {chi_goodness_of_fit_result.statistic}
P-value: {chi_goodness_of_fit_result.pvalue}
Failed to reject the null hypothesis
=> {col} is representative of the population at alpha={significance_level}"""
    print(goodness_of_fit_result)


def chi_independence_test(data, col, target, chi2, p, significance_level):
    independence_null_hypothesis = f'{col} and {target} are independent of each other'
    if p <= significance_level:
        independence_result = "Yes"
    else:
        independence_result = "No"

    data = data.append(
        {
        "Independent Variable": col,
        "Chi-square": chi2,
        "P-value": p,
        "Null Hypothesis": independence_null_hypothesis,
        f"Reject Null Hypothesis at alpha={significance_level}?": independence_result
        },
        ignore_index=True
    )
    return data


def split_data(X, y, test_size, random_state=None, oversampling=True):
    if random_state != None:
        if oversampling:
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            X_test, y_test = SMOTE().fit_resample(X_test, y_test)
            return X_train, X_test, y_train, y_test
        else:
            return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, stratify=y)


# def get_mean_accuracy_score(
#         X,
#         y,
#         classifier,
#         test_size,
#         n,
#         transformer=None,
#         decomposer=None):
    
#     scores = []
#     for i in range(n):
#         try:
#             X_train, X_test, y_train, y_test = \
#                 train_test_split(X, y, test_size=test_size, stratify=y)
#             X_train, y_train = SMOTE().fit_resample(X_train, y_train)
#             X_test, y_test = SMOTE().fit_resample(X_test, y_test)
            
#             if transformer!=None:
#                 X_transformed_train = transformer.fit_transform(X_train, y_train)
#                 X_transformed_test = transformer.transform(X_test)

#                 if decomposer!=None:
#                     try:
#                         X_decomposed_train = decomposer.fit_transform(X_transformed_train)
#                         X_decomposed_test = decomposer.transform(X_transformed_test)
#                     except ValueError:
#                         print(f"Decomposition by {decomposer} failed. Moving on to the next iteration.")
                
#                     classifier.fit(X_decomposed_train, y_train)
#                     score = accuracy_score(y_test, classifier.predict(X_decomposed_test))
                
#                 else:
#                     classifier.fit(X_transformed_train, y_train)
#                     score = accuracy_score(y_test, classifier.predict(X_transformed_test))

#             else:
#                 classifier.fit(X_train, y_train)
#                 score = accuracy_score(y_test, classifier.predict(X_test))

#             scores.append(score)
    
#         except Exception as e:
#             print(f"Exception: {e}")
#             if classifier!=None:
#                 print(f"Classifier: {classifier}")
#             if transformer!=None:
#                 print(f"Transformer: {transformer}")
#             if decomposer!=None:
#                 print(f"Decomposer: {decomposer}")
#             print("Moving on.\n")

#     if transformer==None and decomposer==None:
#         print(f"Mean prediction accuracy score based on training with {list(X.columns)}: {round(np.mean(scores), 2)}")
#     return scores


def update_dicts(clf, X, y, n_splits, n, eval_dict, scores_dict, key, random_state):
    start_time = time.time()
    scores = cross_val_score(clf, X, y, scoring='roc_auc',
                             cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n, random_state=random_state))
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    
    eval_dict[key] = {
        "Mean": round(np.mean(scores), 4),
        "Std": round(np.std(scores), 4),
        "Max": round(np.max(scores), 4),
        "Min": round(np.min(scores), 4),
        "Time elapsed": time_elapsed}
    scores_dict[key] = scores

    print(f"Classifier: {key}")
    print(eval_dict[key], "\n")

    return eval_dict, scores_dict


def get_dicts(X, y, n_splits, n, random_state,
              classifiers=None, classifier=None,
              transformers=None, transformer=None,
              decomposers=None, decomposer=None,
              combinations=None):
    eval_dict = {}
    scores_dict = {}
    if classifiers != None:
        for classifier in classifiers:
            key = type(classifier).__name__
            clf = make_pipeline(classifier)
            eval_dict, scores_dict = update_dicts(clf, X, y, n_splits, n,
                                                  eval_dict, scores_dict, key, random_state)

    elif combinations != None:
        if classifier == None:
            raise Exception("You need to specify a classifier!")
        for features in combinations:
            eval_dict, scores_dict = update_dicts(classifier, X[features], y, n_splits, n,
                                                  eval_dict, scores_dict, ", ".join(features), random_state)
            
    elif transformers != None:
        if classifier == None:
            raise Exception("You need to specify a classifier!")
        for transformer in transformers:
            clf = make_pipeline(transformer, classifier)
            eval_dict, scores_dict = update_dicts(clf, X, y, n_splits, n,
                                                  eval_dict, scores_dict, transformer, random_state)

    elif decomposers != None:
        if classifier == None:
            raise Exception("You need to specify a classifier!")
        for decomposer in decomposers:
            clf = make_pipeline(decomposer, classifier)
            eval_dict, scores_dict = update_dicts(clf, X, y, n_splits, n,
                                                  eval_dict, scores_dict, decomposer, random_state)
            
    else:
        raise Exception("Failed to determine what to evaluate.")

    return eval_dict, scores_dict


def plot_scores(scores_dict):
    fig, axes = plt.subplots(len(scores_dict.keys()), 1,
                    figsize=(5, len(scores_dict.keys())*3))
    fig.tight_layout(pad=5)
    for i, (function, scores) in enumerate(scores_dict.items()):
        try:
            ax = axes[i]
        except TypeError:
            ax = axes
        ax.text(0.02, 0,
    f"""
    Mean: {round(np.mean(scores), 2)}
    Std: {round(np.std(scores), 2)}
    Max: {round(np.max(scores), 2)}
    Min: {round(np.min(scores), 2)}
    """)
        sns.distplot(scores, ax=ax)
        ax.set_title(function)
        ax.set_xlabel("Accuracy Score")
        ax.set_xbound(0, 1)


def eval_models(X, y, n_splits, n, random_state):
    classifiers = [
        LogisticRegression(random_state=random_state), # One of the most basic classifiers
        RandomForestClassifier(random_state=random_state), # decision tree ensemble model where each tree is built independently
        XGBClassifier(random_state=random_state) # eXtreme Gradient Boosting; decision tree ensemble model where each tree is built one after another
    ]

    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifiers=classifiers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_classifier_str = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_classifier_str} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def eval_feature_combinations(X, y, n_splits, n, random_state, classifier):
    combinations = [
        ["X1", "X3"],
        ["X1", "X5"],
        ["X1", "X6"],
        ["X1", "X3", "X5"],
        ["X1", "X3", "X6"],
        ["X1", "X5", "X6"],
        ["X1", "X3", "X5", "X6"]
    ]
    
    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifier=classifier,
                                       combinations=combinations)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_features = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_features} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def eval_transformers(X, y, n_splits, n, random_state, classifier):
    transformers = [
        Nystroem(random_state=random_state),
        SkewedChi2Sampler(random_state=random_state),
        PolynomialCountSketch(random_state=random_state),
        AdditiveChi2Sampler(),
        RBFSampler(random_state=random_state),
        PolynomialFeatures(),
        SplineTransformer()
    ]
    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifier=classifier,
                                       transformers=transformers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_transformer = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_transformer} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def eval_decomposers(X, y, n_splits, n, random_state, classifier):
    decomposers = [
        # PCA(n_components=0.95, svd_solver='full'),
        PCA(random_state=random_state),
        KernelPCA(random_state=random_state),
        IncrementalPCA(),
        TruncatedSVD(),
        FeatureAgglomeration()
    ]
    
    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifier=classifier,
                                       decomposers=decomposers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_decomposer = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_decomposer} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def tune_xgb_clr(X, y, n_iter=10, random_state=None, print_best=True):
    classifier = XGBClassifier(random_state=random_state, verbosity=0)
    params = {
        "eta": np.arange(0.01, 0.3, 0.01), # Typical final values : 0.01-0.2
        # "gamma": 
        "max_depth": np.arange(1,10,1), # Typical values: 3-10
        "min_child_weight": np.arange(1, 10, 1), # The larger min_child_weight is, the more conservative the algorithm will be.
        "subsample": np.arange(0, 1, 0.1), # Typical values: 0.5-1; Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
        "lambda": np.arange(0, 1, 0.01), # L2 regularization term on weights (analogous to Ridge regression). Increasing this value will make model more conservative.
        "alpha": np.arange(0, 1, 0.01), # L1 regularization term on weights (analogous to Lasso regression). Increasing this value will make model more conservative.
        "tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"] # The tree construction algorithm used in XGBoost.
    }
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
    best_search = RandomizedSearchCV(estimator=classifier,
                                     param_distributions=params,
                                     scoring="accuracy",
                                     n_iter=n_iter, # n_iter=10 by default
                                     random_state=random_state)
    best_search.fit(X, y)
    if print_best:
        print(best_search.best_estimator_, "\n", best_search.best_score_, "\n", best_search.best_params_)
    return best_search