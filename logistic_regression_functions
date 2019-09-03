# A function that does the Logistic Regression for you.
# Import needed functions
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

# Takes the data Frame (basetable)
# The predictors you want it to work on as a list.
# (target) is the 'y' column name, the one you want it to predict as a list.
# (print_coef) True/False print out the coefficients of the predictors.
# (print_acu) True/False print out the acu of the model.
def logistic_regression(basetable, predictors, target, print_coef=False, print_acu=False):

    # Create the logistic regression model
    logreg = linear_model.LogisticRegression()

    # Define x & y then split the data into train & test groups.
    x = basetable[predictors]
    y = basetable[target]

    # Fit the model.
    logreg.fit(x, y)

    # Print out the coefficients and intercept if print_coef is True.
    if print_coef is True:
        coef = logreg.coef_
        intercept = logreg.intercept_
        for p, c in zip(predictors, list(coef[0])):
            print(p + '\t' + str(c))
        print(intercept)

    # Do prediction on test data.
    # predictions gives 2 numbers for each user, the first that the target is 0 the second that the target is 1
    predictions = logreg.predict_proba(x)[:, 1]

    # Check accuracy and print it out.
    auc = roc_auc_score(y, predictions)
    if print_acu is True:
        print("AUC of the model: " + str(round(auc, 2)))

    return auc


# A function that goes though the predictors to find the best one.
# The current_predictors are the predictors you already have.
# candidate_predictors is a list of other predictors you want to try.
# (target) is the 'y' column name, the one you want it to predict as a string.
# Takes the data Frame (basetable)
# The function will return the best one from the list of candidate_predictors.
def next_best(current_predictors, candidate_predictors, target, basetable):
    # Initialise the best auc and variable.
    best_auc = -1
    best_predictors = None

    for v in candidate_predictors:
        auc_v = logistic_regression(basetable, current_predictors + [v], target)

        if auc_v >= best_auc:
            best_auc = auc_v
            best_predictors = v

    return (best_predictors)

# To iterate over all possible predictors:
# Set the current_predictors as an empty list
# And the candidate_predictors as a list of all possible predictors.
# Set the max_number_predictors that you want (default is 5).
def find_candidate_predictors(basetable, current_predictors, candidate_predictors, target, max_number_predictors=5):
    number_iterations = min(max_number_predictors, len(candidate_predictors))
    for i in range(0, number_iterations):
        next_cand = next_best(current_predictors, candidate_predictors, target, basetable)

        current_predictors = current_predictors + [next_cand]
        candidate_predictors.remove(next_cand)

    return current_predictors
