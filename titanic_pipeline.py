# import packages
import pandas as pd
import numpy as np

# Data cleaning and processing
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Model evaluation
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# Plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Importing the data and cleaning it.
# Read in Titanic passenger data as a Pandas DataFrame.
df = pd.read_csv('titanic.csv', index_col='PassengerId')

# Age has null values, fill in (impute) the missing data with the median age.
df['Age'] = df['Age'].fillna(df['Age'].median())

# Change 'Sex' to an int and rename the column, you could also use dummy rows for this.
df['Sex'] = df['Sex'].replace(['male', 'female'], [1, 0])
df = df.rename(columns={'Sex': 'is_male'})

# Clean the Embarked column
df = df.dropna(subset=['Embarked'])
dummy_embarked = pd.get_dummies(df.Embarked, drop_first=True)
df = df.merge(dummy_embarked, left_index=True, right_index=True)

# Define the data we are going to feed into it. Just selecting the columns we want from the dataframe
# 'Feature' Columns
X = df[['Pclass', 'Age', 'is_male', 'SibSp', 'Fare', 'Parch', 'Q', 'S']]
# 'Target' Column.
y = df['Survived']

# Split the data into a train and test set. 80% train, 20% test with a random seed of 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)

# Make pipeline for choosing the best model
# Select models to test.
models = {
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=27),
    'LogisticRegression': LogisticRegression(random_state=27, max_iter=5000),
    'ExtraTreesClassifier': ExtraTreesClassifier(random_state=27),
    'AdaBoostClassifier': AdaBoostClassifier(random_state=27),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=27),
}

# Balance parameters for unbalanced data.
balance_params = [SMOTE(random_state=27, sampling_strategy=1.0),
                  RandomOverSampler(sampling_strategy='minority',
                                    random_state=27),
                  RandomUnderSampler(sampling_strategy='majority',
                                     random_state=27)
                  ]

# Select the different parameter for the different models to try.
params = {
    'DecisionTreeClassifier': [{'classifier__class_weight': ['balanced'],
                                'class_balance': ['passthrough'],
                                'classifier__max_depth': [10, 100],
                                'classifier__min_samples_split': [20, 100],
                                'classifier__max_features': ['auto']},
                               {'classifier__class_weight': [None],
                                'class_balance': balance_params,
                                'classifier__max_depth': [10, 100],
                                'classifier__min_samples_split': [20, 100],
                                'classifier__max_features': ['auto']}],
    'LogisticRegression': [{'classifier__class_weight': ['balanced'],
                            'class_balance': ['passthrough'],
                            'classifier__C': [0.001, 1, 100]},
                            {'classifier__class_weight': [None],
                             'class_balance': balance_params,
                             'classifier__C': [0.001, 1, 100]}],
    'ExtraTreesClassifier': {'classifier__max_depth': [10],
                             'classifier__n_estimators': [500],
                             'classifier__max_features': [None, 'auto'],
                             'classifier__min_samples_split': [100]},
    'AdaBoostClassifier':  {'classifier__n_estimators': [100],
                            'classifier__learning_rate': [0.2, 0.8, 1.0]},
    'GradientBoostingClassifier': {'classifier__n_estimators': [100],
                                   'classifier__learning_rate': [0.2, 0.8, 1.0]}

}

# Functions that take each model and runs then with each of their parameters.
# Then saves the results.
class EstimatorSelectionHelper:

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, x, y=None, **grid_kwargs):
        results = pd.DataFrame(columns=['classifier_name', 'classifier', 'best_params', 'ROC_AUC'])
        for key in self.keys:
            # print('Running GridSearchCV for %s.' % key, flush=True)
            model = self.models[key]
            params = self.params[key]
            # print(params)
            pipeline = Pipeline(steps=[('class_balance', 'passthrough'),
                                       ('classifier', model)])
            grid_search = GridSearchCV(pipeline, params, **grid_kwargs)
            grid_search.fit(x, y)
            self.grid_searches[key] = grid_search
            results_dict = {'classifier_name': key,
                            'classifier': grid_search.best_estimator_,
                            'best_params': grid_search.best_params_,
                            'ROC_AUC': grid_search.best_score_}
            # print(results_dict)
            results = results.append(results_dict, ignore_index=True)
        #     print("Done with Grid Search for %s." % key, flush=True)
        # print('Done.')
        return results

    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame) * [name]
            frames.append(frame)
        df = pd.concat(frames)

        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator'] + columns
        df = df[columns]
        return df

helper = EstimatorSelectionHelper(models, params)
report = helper.fit(X_train, y_train, scoring='roc_auc', n_jobs=-1, verbose=1)
summary = helper.score_summary()
# print(summary)

# From the summary table we can see that the best performing model is:
# Model: GradientBoostingClassifier
# Params: learning_rate= 0.2, n_estimators= 100

# Set this as the final model.
final_model = GradientBoostingClassifier(learning_rate= 0.2, n_estimators= 100)

# Train the model.
final_model.fit(X_train, y_train)
# Get the probability of a person surviving.
# Returns an array with 2 columns for each value. Fist number being predict 0, second predict 1.
final_pred_prob = final_model.predict_proba(X_test)[:,1]

# Getting ROC Curve fit
fpr, tpr, thresholds = roc_curve(y_test, final_pred_prob)

# Plot the ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression', c='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion_matrix
cf_matrix = confusion_matrix(y_test, final_pred_prob > 0.5)

# plot confusion matrix
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
