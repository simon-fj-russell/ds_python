# Linear Regression model using stats models
# First import the modules
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# User made functions
# Use Stats models to train a linear model
# Using OLS -> ordinary least squares
def train_lr_model(x, y):
    # X is the feature column(s) of the dataframe (can be in the format df['Column Title'])
    # y is the target column of the dataframe (can be in the format df['Column Title'])
    x = sm.add_constant(x) #We add the intercept term
    model = sm.OLS(y, x).fit()
    # print out the summary of the model
    print(model.summary())
    return model

# Plot observed and predicted values with respect to feature 'x' in log scale.
# THERE CAN ONLY BE ONE FEATURE COLUMN (at the moment)
def plot_observed_vs_predicted(x, y, pred):
    # X is the feature column(s) of the dataframe (can be in the format df['Column Title'])
    # y is the target column of the dataframe (can be in the format df['Column Title'])
    fig, ax = plt.subplots()
    plt.plot(x, y, label='Target Column')  # Change label to your target column.
    plt.plot(x, pred, label='Regression')
    ax.set(xlabel='X Axis', ylabel='Y Axis')
    ax.set_xscale('log')  # If you need to log one of the axis
    plt.title("Sample Plot")  # Title of the graph
    plt.show()

# Read as ether csv or from clipboard
data = pd.read_csv('data.csv')
data = pd.read_clipboard()

# Check out the header to make sure everything is there.
print(data.columns.values)

# work out the correlation between the feature columns and the target column
correlation = data.corr().loc[['Feature Column 1','Feature Column 2','Feature Column 3',...],['Target Column']]
print(correlation)

# Select the feature with the highest correlation and train the model on it.
lr1 = train_lr_model(data['Feature Column 1'],data['Target Column'])

# There can be more then one feature column, input x in the format data[['Feature Column 1','Feature Column 2','Feature Column 3',...]]
lr2 = train_lr_model(data[['Feature Column 1','Feature Column 2','Feature Column 3',...]],data['Target Column'])

# Now you have the model as lr1, use it to predict the Target Column based on a Feature Column
pred = lr1.predict(sm.add_constant(data['Feature Column']))
print(pred.head(10))

# Once you have the predicted values you can feed them into the plot function to view the results vs actual data
plot_observed_vs_predicted(data['Feature Column'], data['Target Column'], pred)
