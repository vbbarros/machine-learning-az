import os
import pandas
import statsmodels.formula.api as sm
from operator import itemgetter
from statsmodels.tools import add_constant
 
def backwardElimination(input_matrix, output_array, significance_level=0.05):
    data = add_constant(input_matrix)
    candidate_variables = list(data.columns)
 
    # >1 because we've added a 'const' column
    while len(candidate_variables) > 1:
        data = data.loc[:, candidate_variables]
        regressor = sm.OLS(endog=output_array, exog=data).fit()
        worst_index, p_value = max(enumerate(regressor.pvalues), key=itemgetter(1))
        if p_value > significance_level:
            print(f"Eliminating '{candidate_variables[worst_index]}' with p-value {p_value:.2}")
            del candidate_variables[worst_index]
        else:
            print(f"Final variable selection: {candidate_variables[1:]}")
            print(regressor.summary())
            return data.loc[:, candidate_variables[1:]]
 
    print("No significant correlation found for any variables")
    return None
 
DATA_PATH = './02_regression/s05_multiple_linear_regression/50_Startups.csv'
INDEPENDENT_VARIABLES = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
DEPENDENT_VARIABLE = ['Profit']
SIGNIFICANCE_LEVEL = 0.05
 
dataset = pandas.read_csv(DATA_PATH)
input_matrix = dataset.loc[:, INDEPENDENT_VARIABLES]
output_array = dataset.loc[:, DEPENDENT_VARIABLE]
 
# Transform categorical variables into dummies
input_matrix = pandas.get_dummies(input_matrix, drop_first=True)
 
result = backwardElimination(input_matrix, output_array, significance_level=SIGNIFICANCE_LEVEL)