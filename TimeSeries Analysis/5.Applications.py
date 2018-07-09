#*****************************************************************************
# Heating Oil Vs Natural Gas (Part 1)
print('#*****************************************************************************\
     \n # Heating Oil Vs Natural Gas')

import matplotlib.pyplot as plt
import pandas as pd

HO = pd.read_csv('HO', parse_dates=['Date'], index_col=['Date'])
NG = pd.read_csv('NG', parse_dates=['Date'], index_col=['Date'])


# Plot the prices separately
plt.subplot(2, 1, 1)
plt.plot(7.25*HO, label='Heating Oil')
plt.plot(NG, label='Natural Gas')
plt.legend(loc='best', fontsize='small')

# Plot the spread
plt.subplot(2, 1, 2)
plt.plot(7.25*HO - NG, label='Spread')
plt.legend(loc='best', fontsize='small')
plt.axhline(y=0, linestyle='--', color='k')
plt.show()


#*****************************************************************************
# Heating Oil Vs Natural Gas (Part 2)
print('\n\n#*****************************************************************************\
     \n # Heating Oil Vs Natural Gas (Part 2)')



# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Compute the ADF for HO and NG
result_HO = adfuller(HO['Close'])
print("The p-value for the ADF test on HO: ", result_HO[1])
result_NG = adfuller(NG['Close'])
print("The p-value for the ADF test on NG: ", result_NG[1])

# Compute the ADF of the spread
result_spread = adfuller(7.25 * HO['Close'] - NG['Close'])
print("The p-value for the ADF test on the HO & NG spread: ", result_spread[1])


#*****************************************************************************
# Are Bitcoin and Ethereum Cointegrated?
print('\n\n#*****************************************************************************\
     \n # Are Bitcoin and Ethereum Cointegrated?')


# Cointegration involves two steps:
#     1) regressing one time series on the other to get the cointegration vector
#     2) perform an ADF test on the residuals of the regression

# Regress the value of one crytocurrency, bitcoin (BTC), on another cryptocurrency, ethereum (ETH). 
# If we call the regression coeffiecient b, then the cointegration vector is simply (1,−b).
# Then perform the ADF test on BTC −b ETH



# Import the statsmodels module for regression and the adfuller function
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

BTC = pd.read_csv('BTC', index_col=['Date'], parse_dates=['Date'])
ETH = pd.read_csv('ETH', index_col=['Date'], parse_dates=['Date'])

# Import the statsmodels module for regression and the adfuller function
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# Plot the prices separately
plt.subplot(2, 1, 1)
plt.plot(BTC, label='BTC')
plt.title('Cryptocurrency: Bitcoin')
plt.subplot(2, 1, 2)
plt.plot(ETH, label='ETH')
plt.title('Cryptocurrency: Ethereum')
plt.show()

# Regress BTC on ETH
ETH = sm.add_constant(ETH)
result = sm.OLS(BTC, ETH).fit()

# Compute ADF
b = result.params[1]
adf_stats = adfuller(BTC['Price'] - b*ETH['Price'])
print("The p-value for the ADF test is ", adf_stats[1])


#*****************************************************************************
#Temperature a Random Walk (with Drift)?
print('\n\n#*****************************************************************************\
     \n #Temperature: A Random Walk (with Drift)?')

# Import the adfuller function from the statsmodels module
from statsmodels.tsa.stattools import adfuller

temp_NY = pd.read_fwf('temp_NY', colspecs=[(0, 4), (6, 10)], parse_dates=['DATE'], index_col=['DATE'])
# Convert the index to a datetime object
temp_NY.index = pd.to_datetime(temp_NY.index, format='%Y')

# Plot average temperatures
temp_NY.plot()
plt.title('Temperature Trend with seasonal pattern: \nA Random Walk (with Drift)')
plt.show()

# Compute and print ADF p-value
result = adfuller(temp_NY['TAVG'])
print("The p-value for the ADF test is ", result[1])

#*****************************************************************************
# Getting "Warmed" Up: Look at Autocorrelations?
print('\n\n#*****************************************************************************\
     \n # ACF Vs PACF of Temperature Change')


# Import the modules for plotting the sample ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Take first difference of the temperature Series
chg_temp = temp_NY.diff()
chg_temp = chg_temp.dropna()

# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)

# Plot the ACF
plot_acf(chg_temp, lags=20, ax=axes[0])
axes[0].set_title('ACF of Temperature change')

# Plot the PACF
plot_pacf(chg_temp, lags=20, ax=axes[1])
axes[1].set_title('PACF of Temperature change')
plt.show()


#*****************************************************************************
# Which ARMA Model is Best?
print('\n\n#*****************************************************************************\
     \n # Which ARMA Model is Best?\n')


# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(1) model and print AIC:
mod1 = ARMA(chg_temp, order=(1,0))
res1 = mod1.fit()


# Fit the data to an AR(2) model and print AIC:
mod2 = ARMA(chg_temp, order=(2,0))
res2 = mod2.fit()


# Fit the data to an MA(1) model and print AIC:
mod3 = ARMA(chg_temp, order=(0, 1))
res3 = mod3.fit()


# Fit the data to an ARMA(1,1) model and print AIC:
mod4 = ARMA(chg_temp, order=(1, 1))
res4 = mod4.fit()
print("The AIC for an AR(1) is: ", res1.aic)
print("The AIC for an AR(2) is: ", res2.aic)
print("The AIC for an MA(1) is: ", res3.aic)
print("The AIC for an ARMA(1,1) is: ", res4.aic)

# res1 = []
# for i in range(2):
#     for j in range(2):
#         mod = ARMA(chg_temp, order=(i, j))
#         res1.append(mod.fit())
#
# for i in range(len(res1)):
#     print("The AIC for an ARMA is: ", res1[i].aic)



#*****************************************************************************
# ARMA(1,1,1) model
print('\n\n#*****************************************************************************\
     \n # ARMA(1,1,1) model')

# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima_model import ARIMA

print(temp_NY.tail())
# Forecast interest rates using an AR(1) model
mod = ARIMA(temp_NY, order=(1,1,1))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start='1872-01-01', end='2046-01-01')
plt.legend(loc='best', fontsize='small')
plt.title('Temperature in NY with ARMA(1,1,1) model')
plt.show()

MSFT = pd.read_csv('MSFT', parse_dates=['Date'], index_col=['Date'])
# Forecast interest rates using an AR(1) model
mod = ARIMA(MSFT, order=(1,1,1))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start='2012-08-19', end='2018-01-01')
plt.legend(loc='best', fontsize='small')
plt.title('MSFT stock price with ARMA(1,1,1) model')
plt.show()







