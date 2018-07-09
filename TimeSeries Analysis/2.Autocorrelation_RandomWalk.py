#*****************************************************************************
# Compute the ACF
print('#*****************************************************************************\
     \n#Compute the ACF')

# Import the acf module and the plot_acf module from statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


HRB = pd.read_csv('HRB', parse_dates=['Quarter'], index_col=['Quarter'])
# Compute the acf array of HRB
acf_array = acf(HRB)
print('AutoCorrelation earnings of H&R block\n'+str(acf_array))

# Plot the acf function
print('ACF plot of H&R block')
plot_acf(HRB, alpha=1)
plt.title('ACF plot of H&R block')
plt.xlabel('lags')
plt.show()

#*****************************************************************************
# ACF N-Lag
print('\n#*****************************************************************************\
     \n# ACF N-Lag')


MSFT = pd.read_csv('MSFT', parse_dates=['Date'], index_col=['Date'])
MSFT['adj_close_chnge'] = MSFT['Adj_Close'].pct_change()
MSFT = MSFT.dropna()
print(MSFT.head())
# Compute and print the autocorrelation of MSFT weekly returns
autocorrelation = MSFT['adj_close_chnge'].autocorr()
print("The autocorrelation of weekly MSFT returns is %4.2f" %(autocorrelation))

# Find the number of observations by taking the length of the returns DataFrame
nobs = len(MSFT)

# Compute the approximate confidence interval
conf = 1.96/pow(nobs, 1/2)
print("The approximate confidence interval is +/- %4.2f" %(conf))

# Plot the autocorrelation function with 95% confidence intervals and 20 lags using plot_acf
print('ACF of MSFT with 95% confidence intervals and 20 lags')
plot_acf(MSFT['adj_close_chnge'], alpha=0.05, lags=20)
plt.title('ACF of MSFT with 95% confidence intervals and 20 lags')
plt.xlabel('lags')
plt.show()


#*****************************************************************************
# White Noise
print('\n#*****************************************************************************\
     \n # White Noise')

# Simulate wite noise returns
print('Simulating White Noise...')
returns = np.random.normal(loc=0.02, scale=0.05, size=1000)

# Print out the mean and standard deviation of returns
mean = np.mean(returns)
std = np.std(returns)
print("The mean is %5.3f and the standard deviation is %5.3f" %(mean,std))

# Plot returns series
print('Plot: White Noise Series')
plt.plot(returns)
plt.title("Simulated White Noise Series\n: mean = %5.3f, stddev = %5.3f" %(mean,std))

# Plot autocorrelation function of white noise returns
print('Plot: ACF of white noise')
plot_acf(returns, lags=20)
plt.title('ACF of Simulated White Noise Series')
plt.show()


#*****************************************************************************
# Creating Random Walk
print('\n#*****************************************************************************\
     \n # Creating Random Walk')

# Generate 500 random steps with mean=0 and standard deviation=1
steps = np.random.normal(loc=0, scale=1, size=500)

# Set first element to 0 so that the first price will be the starting stock price
steps[0]=0

# Simulate stock prices, P with a starting price of 100
print('Simulate stock prices, start price =100, mean=0, stddev=1, steps=500, type=cumulative')
P = 100 + np.cumsum(steps)

# Plot the simulated stock prices
print('PLOT Random Walk: Simulated stock prices')
plt.plot(P)
plt.title("Random Walk: Simulated stock prices\nstart price =100, mean=0, stddev=1, steps=500, type=cumulative")
plt.show()


#*****************************************************************************
# Creating Random Walk with Drift
print('\n#*****************************************************************************\
     \n # Creating Random Walk with Drift')

# Generate 500 random steps
steps = np.random.normal(loc=0.001, scale=0.01, size=500) + 1

# Set first element to 1
steps[0]=1

# Simulate the stock price, P, by taking the cumulative product
print('Simulate stock prices, start price =100, mean=0, stddev=1, steps=500, type=cumulative, drift=1')
P = 100 * np.cumprod(steps)

# Plot the simulated stock prices
print('PLOT Random Walk with Drift: Simulated stock prices')
plt.plot(P)
plt.title("Random Walk with Drift: Simulated stock prices\nstart price =100, mean=0, stddev=1, steps=500, type=cumulative, drift=1")
plt.show()


#*****************************************************************************
# Stock Prices follow a Random Walk
print('\n#*****************************************************************************\
     \n # Stock Prices follow a Random Walk')


# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# MSFT = pd.read_csv('MSFT', parse_dates=['Date'], index_col=['Date'])
# Run the ADF test on the price series and print out the results
results = adfuller(MSFT['Adj_Close'])
print('Augmented Dickey-Fuller test of MSFT Adj Close price:')
print(results)

# Just print out the p-value
### results[0] is the test statistic, and results[1] is the p-value
print('\nThe p-value of the test on stock prices is: ' + str(results[1]))

# Stock Returns
returns = adfuller(MSFT['adj_close_chnge'])
print('\nAugmented Dickey-Fuller test of MSFT Adj Close change:')
print(returns)
### results[0] is the test statistic, and results[1] is the p-value
print('\nThe p-value of the test on prices returns is: ' + str(returns[1]))


#*****************************************************************************
# Seasonal Adjustment in time series
print('\n#*****************************************************************************\
     \n # Seasonal Adjustment in time series')

# Plot ACF of HRB
print('Plot: ACF of H&R block')
plot_acf(HRB)
plt.title('ACF of H&R block')

# Seasonally adjust quarterly earnings
HRB['Earnings_Adjstd_quarterly'] = HRB.diff(4)
# Drop the NaN data in the first three three rows
HRBsa = HRB.dropna()
# Print the first 10 rows of the seasonally adjusted series
print(HRBsa.head(10))

# Plot the autocorrelation function of the seasonally adjusted series
print('Plot: ACF of H&R block with adjusted quarterly earnings')
plot_acf(HRBsa['Earnings_Adjstd_quarterly'])
plt.title('ACF of H&R block with adjusted quarterly earnings')
plt.show()

