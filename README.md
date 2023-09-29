# Happy Customers

### <b>Background</b>

We are one of the fastest growing startups in the logistics and delivery domain. We work with several partners and make on-demand delivery to our customers. During the COVID-19 pandemic, we are facing several different challenges and everyday we are trying to address these challenges.

We thrive on making our customers happy. As a growing startup, with a global expansion strategy we know that we need to make our customers happy and the only way to do that is to measure how happy each customer is. If we can predict what makes our customers happy or unhappy, we can then take necessary actions.

Getting feedback from customers is not easy either, but we do our best to get constant feedback from our customers. This is a crucial function to improve our operations across all levels.

We recently did a survey to a select customer cohort. You are presented with a subset of this data. We will be using the remaining data as a private test set.

### <b>Data Description</b>

- Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers<br>
- X1 = my order was delivered on time<br>
- X2 = contents of my order was as I expected<br>
- X3 = I ordered everything I wanted to order<br>
- X4 = I paid a good price for my order<br>
- X5 = I am satisfied with my courier<br>
- X6 = the app makes ordering easy for me<br>

Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5 where the smaller number indicates less and the higher number indicates more towards the answer.

### <b>Objectives</b>
- Predict if a customer is happy or not based on the answers they give to questions asked, and reach 73% accuracy score or above.
- Using a feature selection approach, identify the minimal set of attributes/features that would preserve the most information about the problem while increasing predictability of the data we have.
- Using a feature selection approach, identify any question that we can remove in our next survey.

### <b> Results</b>

<u>Model Performance</u>

We developed a Logistic Regression model primarily for its efficiency since we needed to explore diverse approaches for optimal results. Simpler approaches failed to yield a good prediction accuracy score mainly due to limited data for modeling, which only consists of 126 data points, or customers. To achieve a prediction accuracy of 73% or higher, we executed the following key processes:

- Feature analysis and selection: Conducted a comprehensive analysis of features and carefully selected relevant ones for the model. More details on this process are outlined in the subsequent item.
- Handling class imbalance: Addressed class imbalance using SMOTE (synthetic minority oversampling technique) to ensure a more balanced representation of the classes.
- Feature augmentation: Implemented various feature augmentation techniques to overcome limitations posed by a restricted number of features during modeling and evaluation. Notably, the application of the RBF Sampler from scikit-learn enhanced the mean accuracy score the most, up to 0.6815 when combined with the Logistic Regression model.
- Hyperparameter Tuning: Employed RandomizedSearchCV from scikit-learn to fine-tune the hyperparameters of the Logistic Regression model.

After thorough experimentation and meticulous examination, our model successfully achieved the targeted accuracy score.

<u>Critical Features</u>

Our objective was to pinpoint the optimal set of features that retain the most pertinent information for predicting customer happiness while enhancing predictive capacity of our data and model. To achieve this, we rigorously analyzed and employed various feature selection methods, primarily including the following:

- Chi-square test of goodness of fit: Utilized to assess whether the sample data is a representative subset of the overall population. The null hypothesis was that there is no significant difference between a variable and its expected frequencies. We failed to reject the null hypothesis for all 6 dependent variables, implying their representativeness within the population at a significance level (alpha) of 0.05.
- Chi-square test of independence: Employed to evaluate if differences between observed and expected data are due to chance. The null hypothesis was that a dependent variable (X#) and the target variable (Y) are independent. At a significance level of 0.05, we rejected the null hypothesis only for X1. With a higher alpha of 0.1, we would also reject the null hypothesis for X6, indicating that X6 and Y are not independent. In summary, X2-X5 are independent of Y, rendering them less useful in predicting Y. Meanwhile, X1 and, to a lesser extent, X6 would aid in predicting Y.
- Exploration of different feature combinations: We systematically explored various feature combinations to identify the most effective set.

The combination of X1 and X3 yielded the highest mean prediction accuracy score, making them the chosen features for subsequent modeling and evaluation. This result suggests that, in future iterations, other features could potentially be omitted from surveys, as they were found to have lesser impact on predicting customer satisfaction. Additionally, gathering data from a larger customer pool would likely enhance model performance, considering our model was developed with a limited dataset - a primary challenge encountered during this project.

### <b>Notebook and Installation</b>

For more details, you may refer to <a href='https://github.com/henryhyunwookim/Happy-Customers/blob/main/Happy%20Costomers.ipynb'>this notebook</a> directly.

To run Happy Costomers.ipynb locally, please clone or fork this repo and install the required packages by running the following command:

pip install -r requirements.txt

You can also view the notebook saved as PDF (Happy Costomers - Jupyter Notebook.pdf) in this repository without having to install anything.

##### Source: Apziva
