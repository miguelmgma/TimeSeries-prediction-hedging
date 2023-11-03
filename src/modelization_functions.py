
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from scipy.stats import kurtosis, skew
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

def stepwise_regression(X, y,n_vars:int = 4):
    '''
    It performs the step wise method for the variables entered in the dataframe X. It is delimited with the parameter n_vars.
    The step-wise method in regression is a systematic approach to select the most relevant variables and build an optimal regression mode.
    If no variables are found relevant, it defaults to the Pool Avg variable
    Input:
        X: Dataframe. Dataframe with all coluns to performa the Step Wise regression
        y: Array. Array constaining the objective variable values 
        n_cols: Integer. The maximum number of columns to select

    Output: List.
        It outputs a list containing the name of the relevant columns found with 2 particular rules applied:
            When no column is found significant, it defaults to the Pool Avg column
            When more than n_vars variables are found, it takes only the n_vars more relevant columns
    '''
    included = list(X.columns)
    while True:
        model = sm.OLS(y, sm.add_constant(X[included])).fit()

        max_pval = model.pvalues[1:].max()  # Excluye el p-valor del intercepto


        if max_pval > 0.05:
            excluded_feature = model.pvalues[1:].idxmax()
            included.remove(excluded_feature)
        else:
            pvals = model.pvalues[1:].sort_values(ascending = False)
            break
    if len(included) == 0:
        included = ['POOL AVG']
    elif len(included) > (n_vars):
        vars_to_rmv = len(included) - n_vars
        included = list(pvals.index[vars_to_rmv:])
    else:
        pass

    return included
    
    
def regression(dataframe: pd.DataFrame(), variables:list, y_var : str , df_test = None,reg_type:str = 'Linear',significativas:bool = False, combs:bool = False,n_vars:int = 4):
    '''
    It performs the regression based on the dataframe passed. It outputs a dataframe with all metrics and data relevant to the regression.

    Input:
        dataframe: Dataframe. Dataframe that contains all variables, the objective function
        variables: List. List constaining the objective variable values 
        y_var: String. The objective variable  of the regression
        df_test: Dataframe. It is the test dataframe in which the regression is evaluated
        reg_type: String. Linear for linear regression or Huber for robust regression
        significativas: Boolean. True to apply the Step Wise method in the regression. False to select all variales passed in x_cols.
        combs: Boolean. It calculates the regression for each combination of variables, with a maximum of n_vars 
        n_vars: Integer. Maximum number of variables passed to the regression. It is used when using the combinations (The parameter combs must be True) and when using Step Wise regression (The parameter signif must be True)

    Output: Dataframe.
        It outputs a dataframe with multiple columns:
            Num variables: Number of variables used in the regression
            Variables: String with the concatenation of the variables used in the regression
            vars: List of the variables used in the regression
            formula: String with the formula used in the regression
            coef: Coefficients of the variables used in the regression
            intercept: Intercept of the regression
            r2: R2 of the regression
            adj_r2: Adjusted R2 of the regression
            MSE:  Mean Square Error of the regression
            MAE:  Mean Average Error of the regression
            MAPE: Mean Average Percentage Error of the regression
            skewness: skewness of the regression
            kurtosis: kurtosis of the regression
            perc_80: Quantile 0.8 of the residuals of the regression
            perc_95: Quantile 0.95 of the residuals of the regression
            Cointegrados: If the residuals are cointegrated. 1 if yes, 0 if no 
            Estructura: If the residuals has structure. 1 if yes, 0 if no 
            test: Objective variable real values of the test dataframe 
            pred: Objective variable predictions 
            residuos: Residuals. Objective variable real values minus its predictions
    '''
    resultados = []
    combinaciones = []
    combinaciones_filtradas = []
    res_test = []
    res_pred = []

    if combs:
        max_variables = min(len(variables), n_vars)
        for r in range(3, max_variables + 1):

            combinaciones.extend(itertools.combinations(variables, r))

        for combinacion in combinaciones:
            has_similar_name = False
            if len(set([var[:4] for var in combinacion])) == len(combinacion):
                combinaciones_filtradas.append(combinacion)
    else: 
        combinaciones_filtradas = list([variables])
    

    for combinacion in combinaciones_filtradas:
        variables_comb =  list(combinacion)
        x = dataframe[variables_comb]
        y = dataframe[y_var]
        x_test_in_sample = x.copy()
        y_test_in_sample = y.copy()

        if df_test is None:
            x_test = x.copy()
            y_test = y.copy()
        else:                               # IF we are using out-of-sample bt
            x_test = df_test[variables_comb]
            try:
                y_test = df_test[y_var]
            except:
                y_test = pd.Series( [0] * len(df_test))

        if significativas: # Apply Step Wise Regression

            variables_comb = stepwise_regression(x,y,n_vars)
            
            x = x[variables_comb]
            x_test = x_test[variables_comb]
            x_test_in_sample = x[variables_comb].copy()

        if reg_type == 'Huber': # Robust regression

            model = HuberRegressor()
            result = model.fit(x,y)

            coeficientes = model.coef_
            intercept = model.intercept_

            y_pred_in_sample = result.predict(x_test_in_sample)
            y_pred = result.predict(x_test)
            r2 = r2_score(y_test_in_sample, y_pred_in_sample)# get the r2 in-sample

            
        if reg_type == 'Linear': # Linear regression
             
            model = LinearRegression()
            result = model.fit(x,y)

            coeficientes = model.coef_  
            intercept = model.intercept_
            
            y_pred = result.predict(x_test)
            y_pred_in_sample = result.predict(x_test_in_sample)
            r2 = r2_score(y_test_in_sample, y_pred_in_sample) # get the r2 in-sample
        
        # Get the formula

        formula = f" Y = {round(intercept,3)}"

        for i, coef in enumerate(coeficientes):
            formula += f" + {round(coef,3)} * {variables_comb[i]}"

        n = len(y_test) #número de observaciones
        p = len(variables_comb) #número de predictores (p): Multivariante
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        if  '_log' in y_var: # If the columns are logs, we have to retransform the results
            
            y_test = np.exp(y_test)
            y_pred = np.exp(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) 
        
        # Residuals analysis

        residuos = y_test - y_pred
        
        skewness = skew(residuos)
        kurt = kurtosis(residuos)
        perc_80 = residuos.quantile(0.8)
        perc_95 = residuos.quantile(0.95)

        if df_test is None:
            res_adf = adfuller(residuos)
            if res_adf[1] < 0.05:
                co_int = 1
            else:
                co_int = 0

            lb = acorr_ljungbox(residuos)
            if any(float(p) < 0.05 for p in lb['lb_pvalue']):
                estruc = 1
            else:
                estruc = 0

        else:
            co_int = 0
            estruc = 0

        #   Dataframe creation

        res_test.append(y_test)
        res_test1 = np.concatenate(res_test).tolist()
        res_test1 = np.concatenate([np.ravel(rr) for rr in res_test1])

        res_pred.append(y_pred)
        res_pred1 = np.concatenate(res_pred).tolist()
        res_pred1 = np.concatenate([np.ravel(rr) for rr in res_pred1])

        resultados.append((len(variables_comb),', '.join(variables_comb), variables_comb, formula, coeficientes,intercept,r2, adj_r2,mse, mae, mape,skewness, kurt, perc_80,perc_95,  co_int, estruc,res_test1,res_pred1, np.array(residuos)))
    
    df_resultados = pd.DataFrame(resultados, columns=['Num Variables', 'Variables','vars','formula','coef','intercept', 'r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','Cointegrados', 'Estructura','test','pred','residuos'])

    return df_resultados