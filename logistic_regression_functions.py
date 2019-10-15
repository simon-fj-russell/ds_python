# Linear Regression model using stats models
# First import the modules
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Check that it worked.
print(df.info())

# Or you can use an apply lambda function (like an if statement).
# df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# Check the correlation between the “feature” columns and the “Target” column.
correlation = df.corr().loc[['Pclass', 'Age', 'is_male', 'SibSp', 'Fare', 'Parch'], ['Survived']]
print(correlation)

# Select the model we want to use
logreg = LogisticRegression(solver='lbfgs')

# Define the data we are going to feed into it. Just selecting the columns we want from the dataframe
# 'Feature' Columns
X = df[['Pclass', 'Age', 'is_male', 'SibSp', 'Fare', 'Parch']]
# Instead of selecting the columns we want, we can drop columns. Using axis=1 to selects columns not rows
# X = df.drop('Survived', axis=1)
# 'Target' Column.
y = df[['Survived']]

# Split the data into a train and test set. 80% train, 20% test with a random seed of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fitting the model
logreg.fit(X_train, y_train)

# Now check the accuracy of the model using the test data.
ml_score = logreg.score(X_test, y_test)
print("Accuracy of model =", round(ml_score, 4))
