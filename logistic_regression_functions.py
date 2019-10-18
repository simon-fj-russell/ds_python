# Linear Regression model using stats models
# First import the modules
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Read in Titanic passenger data as a Pandas DataFrame.
df = pd.read_csv('titanic.csv')

# Also make a raw data that we won't change.
df_raw = pd.read_csv('titanic.csv')

# Look at dataframe, column titles and missing data.
# df.describe() also give a very detailed breakdown of the data in the column
print(df.info())

# Age has null values, fill in the missing data with the median age.
df['Age'] = df['Age'].fillna(df['Age'].median())

# Change 'Sex' to an int and rename the column.
df['Sex'] = df['Sex'].replace(['male', 'female'], [1, 0])
df = df.rename(columns={'Sex': 'is_male'})

# Or you can use an apply lambda function (like an if statement).
# df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# Check that it worked.
print(df.info())

# Check the correlation between the “feature” columns and the “Target” column.
correlation = df.corr().loc[['Pclass', 'Age', 'is_male', 'SibSp', 'Fare', 'Parch'], ['Survived']]
print(correlation)

# Select the model we want to use
logreg = LogisticRegression(solver='lbfgs')

# Define the data we are going to feed into it. Just selecting the columns we want from the dataframe
# 'Feature' Columns
X = df[['Pclass', 'Age', 'is_male', 'SibSp', 'Fare', 'Parch']]
# 'Target' Column.
y = df['Survived']

# Split the data into a train and test set. 80% train, 20% test with a random seed of 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fitting the model
logreg.fit(X_train, y_train)

# Now check the accuracy of the model using the test data.
ml_score = logreg.score(X_test, y_test)
print("")
print("Accuracy of model =", round(ml_score, 4))

# For Logistic Regression we can look at the ROC curve.
# Returns an array with 2 columns for each value. Fist number being predict 0, second predict 1.
# Select the second value.
y_pred_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# fpr=false positive rate
# tpr=true positive rate

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# The area under the ROC curve the better the model, AUC.
print("ROC AUC score:",round(roc_auc_score(y_test,y_pred_prob),4))
