#*****************************************************************************
# Simulate AR(1) timeseries
print('#*****************************************************************************\
     \n # Simulate AR(1) timeseries')

# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np

# Plot 1: AR parameter = +0.9
plt.subplot(3, 1, 1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
print('Simulate AR timeseries : parameter = +0.9')
plt.title('Simulated AR timeseries : parameter = +0.9')
plt.plot(simulated_data_1)

# Plot 2: AR parameter = -0.9
plt.subplot(3, 1, 2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
print('Simulate AR timeseries : parameter = -0.9')
plt.title('Simulated AR timeseries : parameter = -0.9')
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)

# Plot 3: AR parameter = +0.3
plt.subplot(3, 1, 3)
ar3 = np.array([1, -0.3])
ma3 = np.array([1])
AR_object3 = ArmaProcess(ar3, ma3)
print('Simulate AR timeseries : parameter = +0..')
plt.title('Simulated AR timeseries : parameter = 0.3')
simulated_data_3 = AR_object3.generate_sample(nsample=1000)
plt.plot(simulated_data_3)
plt.show()

#*****************************************************************************
# Compare ACFs
print('\n\n#*****************************************************************************\
     \n # Compare ACFs')

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(3,1)
# Plot 1: AR parameter = +0.9
print('ACF of AR timeseries : parameter = +0.9')
fig = plot_acf(simulated_data_1, alpha=1, lags=20, ax=axes[0])

# Plot 2: AR parameter = -0.9
print('ACF of AR timeseries : parameter = -0.9')
fig = plot_acf(simulated_data_2, alpha=1, lags=20, ax=axes[1])

# Plot 3: AR parameter = +0.3
print('ACF of AR timeseries : parameter = +0.3')
fig = plot_acf(simulated_data_3, alpha=1, lags=20, ax=axes[2])

# Label axes
axes[0].set_title('ACF of AR timeseries : parameter = +0.9')
axes[1].set_title('ACF of AR timeseries : parameter = -0.9')
axes[2].set_title('ACF of AR timeseries : parameter = +0.3')
plt.show()


#*****************************************************************************
# Estimating the parameters of AR Model
print('\n\n#*****************************************************************************\
     \n # Estimating the parameters of AR Model')


# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Fit an AR(1) model to the first simulated data
print('Fit an AR(1) model to the first simulated data')
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for phi
print("When the true phi=0.9, the estimate of phi (and the constant) are:")
print(res.params)


#*****************************************************************************
# Forecasting with an AR Model
print('\n\n#*****************************************************************************\
     \n # Forecasting with an AR Model')

# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast the first AR(1) model
mod = ARMA(simulated_data_1, order=(1, 0))
res = mod.fit()
print('Forecast with first AR(1) model:\nstart=990, end=1010')
res.plot_predict(start=990, end=1010)
plt.title('Forecast with first AR(1) model:\nstart=990, end=1010')
plt.show()


#*****************************************************************************
# Forecast Interest Rates
print('\n\n#*****************************************************************************\
     \n # Forecast Interest Rates')

import pandas as pd

interest_rate_data = pd.read_csv('interest_rates', parse_dates=['DATE'],index_col=['DATE'])

# Forecast interest rates using an AR(1) model
print('Forecast interest rates using an AR(1) model')
mod = ARMA(interest_rate_data, order=(1,0))
res = mod.fit()

# get in series

fut = res.predict(start='2017', end='2018')
print(fut)

# Plot the original series and the forecasted series
res.plot_predict(start=0, end='2022')
plt.title('Forecast interest rates with AR(1) model:\nstart=1960, end=2022')
plt.legend(fontsize=8)
plt.show()


#*****************************************************************************
# Compare AR Model with Random Walk
print('\n\n#*****************************************************************************\
     \n # Compare AR Model with Random Walk')

simulated_data = np.array([5.,  4.77522278,  5.60354317,  5.96406402,  5.97965372,
        6.02771876,  5.5470751 ,  5.19867084,  5.01867859,  5.50452928, 5.89293842,  4.6220103 ,  5.06137835,  5.33377592,  5.09333293,
        5.37389022,  4.9657092 ,  5.57339283,  5.48431854,  4.68588587, 5.25218625,  4.34800798,  4.34544412,  4.72362568,  4.12582912,
        3.54622069,  3.43999885,  3.77116252,  3.81727011,  4.35256176, 4.13664247,  3.8745768 ,  4.01630403,  3.71276593,  3.55672457,
        3.07062647,  3.45264414,  3.28123729,  3.39193866,  3.02947806, 3.88707349,  4.28776889,  3.47360734,  3.33260631,  3.09729579,
        2.94652178,  3.50079273,  3.61020341,  4.23021143,  3.94289347, 3.58422345,  3.18253962,  3.26132564,  3.19777388,  3.43527681,
        3.37204482])
## Plot the interest rate series and the simulated random walk series side-by-side
print('Plot the interest rate series and the simulated random walk series side-by-side')
plt.subplot(2, 1, 1)
plt.title("Interest Rate Data")
plt.plot(interest_rate_data)
plt.subplot(2, 1, 2)
plt.title("Simulated Random Walk Series")
plt.plot(simulated_data)
plt.show()


# Plot the autocorrelation of the interest rate series in the top plot
fig, axes = plt.subplots(2,1)
print('Plot the autocorrelation of the interest rate series in the top plot')
fig = plot_acf(interest_rate_data, alpha=1, lags=12, ax=axes[0])

# Plot the autocorrelation of the simulated random walk series in the bottom plot
print('Plot the autocorrelation of the simulated random walk series in the bottom plot')
fig = plot_acf(simulated_data, alpha=1, lags=12, ax=axes[1])

# Label axes
axes[0].set_title("Autocorrelation of Interest Rate Data")
axes[1].set_title("Autocorrelation of Simulated Random Walk Series")
plt.show()


#*****************************************************************************
# Estimate Order of Model: PACF
print('\n\n#*****************************************************************************\
     \n # Estimate Order of Model: PACF')

# Import the modules for simulating data and for plotting the PACF
from statsmodels.graphics.tsaplots import plot_pacf
fig, axes = plt.subplots(2,1)
# Simulate AR(1) with phi=+0.6
ma = np.array([1])
ar = np.array([1, -0.6])
AR_object = ArmaProcess(ar, ma)
print('Simulate AR(1) with phi=+0.6')
simulated_data_1 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(1)
fig = plot_pacf(simulated_data_1, lags=20, ax=axes[0])
axes[0].set_title('Plot PACF for AR(1) lag=20 with phi=+0.6')

# Simulate AR(2) with phi1=+0.6, phi2=+0.3
ma = np.array([1])
ar = np.array([1, -0.6, -0.3])
AR_object = ArmaProcess(ar, ma)
print('Simulate AR(2) with phi1=+0.6, phi2=+0.3')
simulated_data_2 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(2)
fig = plot_pacf(simulated_data_2, lags=20, ax=axes[1])
axes[1].set_title('Plot PACF for AR(2) with phi1=+0.6, phi2=+0.3')
plt.show()


#*****************************************************************************
# Estimate Order of Model: Information Criteria
print('\n\n#*****************************************************************************\
     \n # Estimate Order of Model: Information Criteria')

# Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC). 
# These measures compute the goodness of fit with the estimated parameters, 
# but apply a penalty function on the number of parameters in the model.


# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
# print(simulated_data_2)
for p in range(7):
    mod = ARMA(simulated_data_2, order=(p,0))
    res = mod.fit()
# Save BIC for AR(p)
    BIC[p] = res.bic

# Plot the BIC as a function of p
plt.plot(range(1,7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Baysian Information Criterion')
plt.title('BIC plot')
plt.show()


