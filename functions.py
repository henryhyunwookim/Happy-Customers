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


# Define the function for loading the data set
def load_data(file_name, folder_name=None):
    if folder_name != None:
        path = Path("/".join([os.getcwd(), folder_name, file_name]))
    else:
        path = Path("/".join([os.getcwd(), file_name]))
    return pd.read_csv(path)


# decomposers = [
#     # PCA(n_components=0.95, svd_solver='full'),
#     PCA(),
#     KernelPCA(),
#     FastICA(),
#     SparsePCA(),
#     IncrementalPCA(),
#     TruncatedSVD(),
#     MiniBatchSparsePCA(),
#     FeatureAgglomeration()
# ]  


# Define functions
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
                train_test_split(X, y, test_size=test_size, random_state=random_state)
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            X_test, y_test = SMOTE().fit_resample(X_test, y_test)
            return X_train, X_test, y_train, y_test
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size)


def get_mean_accuracy_score(
        X,
        y,
        classifier,
        test_size,
        n,
        transformer=None,
        decomposer=None):
    
    scores = []
    try:
        for i in range(n):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=test_size)
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            X_test, y_test = SMOTE().fit_resample(X_test, y_test)
            
            if transformer==None:
                classifier.fit(X_train, y_train)
                score = accuracy_score(y_test, classifier.predict(X_test))
            
            elif decomposer!=None:
                X_transformed_train = transformer.fit_transform(X_train, y_train)
                X_transformed_test = transformer.transform(X_test)
                
                try:
                    components = decomposer.fit_transform(X_transformed_train)
                except ValueError:
                    print(f"Decomposition by {decomposer} failed. Moving on to the next iteration.")
                
                classifier.fit(components, y_train)
                score = accuracy_score(y_test, classifier.predict(decomposer.transform(X_transformed_test)))
            
            else:
                X_transformed = transformer.fit_transform(X_train, y_train)  
                classifier.fit(X_transformed, y_train)
                score = accuracy_score(y_test, classifier.predict(transformer.transform(X_test)))

            scores.append(score)
    
    except Exception as e:
        print(f"Exception: {e}")
        if classifier!=None:
            print(f"Classifier: {classifier}")
        if transformer!=None:
            print(f"Transformer: {transformer}")
        if decomposer!=None:
            print(f"Decomposer: {decomposer}")
        print("Moving on.\n")

    if transformer==None and decomposer==None:
        print(f"Mean prediction accuracy score based on training with {list(X.columns)}: {round(np.mean(scores), 2)}")
    return scores


def get_dicts(X, y, test_size, n,
              classifiers=None, classifier=None,
              transformers=None, transformer=None,
              decomposers=None, decomposer=None):
    eval_dict = {}
    scores_dict = {}

    try:
        if classifiers!=None:
            for classifier in classifiers:
                key = type(classifier).__name__
                start_time = time.time()
                if transformer==None:
                    scores = get_mean_accuracy_score(X, y, classifier, test_size, n)
                else:
                    scores = get_mean_accuracy_score(X, y, classifier, test_size, n,
                                                    transformer=transformer)
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                
                eval_dict[key] = {
                    "Mean": round(np.mean(scores), 2),
                    "Std": round(np.std(scores), 2),
                    "Max": round(np.max(scores), 2),
                    "Min": round(np.min(scores), 2),
                    "Time elapsed": time_elapsed}
                scores_dict[key] = scores

                print(f"Classifier: {key}")
                print(eval_dict[key], "\n")

        elif transformers!=None:
            for transformer in transformers:
                start_time = time.time()
                scores = get_mean_accuracy_score(X, y, classifier, test_size, n,
                                                 transformer=transformer)
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                
                eval_dict[transformer] = {
                    "Mean": round(np.mean(scores), 2),
                    "Std": round(np.std(scores), 2),
                    "Max": round(np.max(scores), 2),
                    "Min": round(np.min(scores), 2),
                    "Time elapsed": time_elapsed}
                scores_dict[transformer] = scores

                print(f"Transformer: {transformer}")
                print(eval_dict[transformer], "\n")
                
        elif decomposers!=None:
            for decomposer in decomposers:
                start_time = time.time()
                scores = get_mean_accuracy_score(X, y, classifier, test_size, n,
                                                 transformer=transformer,
                                                 decomposer=decomposer)
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                
                eval_dict[decomposer] = {
                    "Mean": round(np.mean(scores), 2),
                    "Std": round(np.std(scores), 2),
                    "Max": round(np.max(scores), 2),
                    "Min": round(np.min(scores), 2),
                    "Time elapsed": time_elapsed}
                scores_dict[decomposer] = scores
                
                print(f"Decomposer: {decomposer}")
                print(eval_dict[decomposer], "\n")

    except Exception as e:
        print(f"Exception: {e}")
        if classifier!=None:
            print(f"Classifier: {classifier}")
        if transformer!=None:
            print(f"Transformer: {transformer}")
        if decomposer!=None:
            print(f"Decomposer: {decomposer}")
        print("Moving on.\n")

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

logistic_regression = LogisticRegression() # One of the most basic classifiers
random_forest = RandomForestClassifier() # decision tree ensemble model where each tree is built independently
xg_boost = XGBClassifier() # eXtreme Gradient Boosting; decision tree ensemble model where each tree is built one after another
ada_boost = AdaBoostClassifier() # Adaptive Boosting
sgd = SGDClassifier() # linear classifier optimized by Stochastic Gradient Descent
mlp = MLPClassifier() # Multi-Layer Perceptron classifier (neural network)
classifiers = [logistic_regression, random_forest, xg_boost, ada_boost, sgd, mlp]
def eval_models(X, y, test_size, n, classifiers=classifiers):
    eval_dict, scores_dict = get_dicts(X, y, test_size, n, classifiers=classifiers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_classifier_str = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_classifier_str} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


combinations = [
        ["X1", "X3"],
        ["X1", "X5"],
        ["X1", "X6"],
        ["X1", "X3", "X5"],
        ["X1", "X3", "X6"],
        ["X1", "X5", "X6"],
        ["X1", "X3", "X5", "X6"]
        ]
def get_best_feature_combination(data,
                                 target,
                                 classifier,
                                 test_size,
                                 n,
                                 combinations=combinations):
    best_score = 0
    for combination in combinations:
        score = np.mean(get_mean_accuracy_score(data[combination],
                                                target,
                                                classifier,
                                                test_size,
                                                n))
        if score > best_score:
            best_score = score
            best_features = ", ".join(combination)
    print(f"\nBest score {round(best_score, 2)} was achieved with {best_features}")
    return best_features


transformers = [
    Nystroem(),
    SkewedChi2Sampler(),
    PolynomialCountSketch(),
    AdditiveChi2Sampler(),
    RBFSampler(),
    PolynomialFeatures(),
    SplineTransformer()
    ]
def eval_transformers(X, y, test_size, n, classifier, transformers=transformers):
    eval_dict, scores_dict = get_dicts(X, y, test_size, n, classifier=classifier, transformers=transformers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_transformer = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_transformer} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


# def get_accuracy_score_with_all_features(transformers=transformers, classifier=classifier, n=n, test_size=test_size):
#     scores = []
#     for i in range(n):
#         X_balanced, y_balanced = SMOTE().fit_resample(survey_df[["X1", "X6"]], survey_df["Y"])
        
#         X_train, X_test, y_train, y_test = \
#             train_test_split(X_balanced, y_balanced, test_size=test_size)
        
#         for j, transformer in enumerate(transformers):
#             X_transformed_train = transformer.fit_transform(X_train, y_train)
#             X_transformed_test = transformer.transform(X_test)
#             if j==0:
#                 X_combined_train = X_transformed_train
#                 X_combined_test = X_transformed_test
#             else:
#                 X_combined_train = np.concatenate((X_combined_train, X_transformed_train), axis=1)
#                 X_combined_test = np.concatenate((X_combined_test, X_transformed_test), axis=1)
        
#         if i == range(n)[0]:
#             print(f"Total number of features for training: {X_combined_train.shape[1]}")
#         classifier.fit(X_combined_train, y_train)
#         score = accuracy_score(y_test, classifier.predict(X_combined_test))
#         scores.append(score)

#     return round(np.mean(scores), 2)


# def eval_decomposers(training_features, classifier=classifier, transformer=transformer, decomposers=decomposers):
#     eval_dict, scores_dict = get_dicts(training_features, decomposers=decomposers)

#     plot_scores(scores_dict)

#     eval_df = pd.DataFrame(eval_dict).T.sort_values(
#         ["Mean", "Max", "Min", "Std", "Time elapsed"],
#         ascending=[False, False, True, True, True])
#     print(f"{eval_df.index[0]} yielded the best mean accuracy score of {eval_df.iloc[0, 0]}")
#     return eval_df