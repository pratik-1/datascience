# Import pandas and plotting modules
from pandas import read_csv as read_csv
from pandas import read_fwf as read_fwf
import matplotlib.pyplot as plt
import pandas as pd
#*************************************************************************************

print('#*****************************************************************************\
     \n # Diet Plot')
diet = pd.read_csv('Diet', parse_dates=['Date'], index_col=['Date'])
#
# # Convert the date index to datetime
diet.index = pd.to_datetime(diet.index)
#
# # Plot 2012 data using slicing
diet['2012'].plot(grid=True)
plt.title('Diet for year 2012')
print('Diet Plot1')
plt.show()

# # Plot the entire time series diet and show gridlines
diet.plot(grid=True)
plt.title('Diet for year 2012 to 2016')
print('Diet Plot2')
plt.show()

#*****************************************************************************
# Merging Time Series With Different Dates
print('#*****************************************************************************\
     \n # Merging Time Series With Different Dates')

stocks = read_csv('stocks', parse_dates=['observation_date'], index_col=['observation_date'])
bonds = read_csv('bonds.csv', parse_dates=['observation_date'], index_col=['observation_date'])

# bonds = pd.read_fwf('bonds', colspecs=[(0,16),(19,24)],index_col=['observation_date'])
# bonds.to_csv('bonds.csv')


# Convert the stock index and bond index into sets
set_stock_dates = set(stocks.index)
set_bond_dates = set(bonds.index)

# Merge stocks and bonds DataFrames using join()
stocks_and_bonds = stocks.join(bonds, how='inner')
print('stocks and bonds\n'+str(stocks_and_bonds.head()))

#*****************************************************************************
# Correlation of Stocks and Bonds
print('#*****************************************************************************\
     \n # Correlation of Stocks and Bonds')

# Compute percent change using pct_change()
returns = stocks_and_bonds.pct_change()
print(returns.head())
# Compute correlation using corr()
correlation = returns['SP500'].corr(returns['US10Y'])
print("Correlation of stocks and interest rates: ", correlation)

# Make scatter plot
plt.scatter(returns['SP500'], returns['US10Y'])
plt.title('SP500 Vs US10Y')
plt.xlabel('SP500')
plt.ylabel('US10Y')
#plt.show()

#*****************************************************************************
# Looking at a Regression's R-Squared
print('#*****************************************************************************\
     \n # Looking at a Regression\'s R-Squared')


import statsmodels.formula.api as sm

x1 = read_csv('x',header=None, names=['rownum','x'])
y1 = read_csv('y',header=None, names=['rownum','y'])
df = x1.merge(y1)
# print(pd.Series(y1, inplace=True))
correlation1 = x1['x'].corr(y1['y'])
print("The correlation between x and y is %4.2f" %(correlation1))
#print(z1.head())
result = sm.ols(formula="y ~ x", data=df).fit()
print('\n\nSummary resuts between x and y\n')
print(result.summary())

#*****************************************************************************
# A Popular Strategy Using Autocorrelation
print('\n#*****************************************************************************\
     \n # A Popular Strategy Using Autocorrelation')

# mean reversion in stock prices: prices tend to bounce back, or revert, towards \
# previous levels after large moves, which are observed over time horizons of about a week.

# Mathematically,
# mean reversion ==  stock returns are negatively autocorrelated
MSFT = read_csv('MSFT', header=0,index_col=['Date'],parse_dates=['Date'])
#Convert the daily data to weekly data
MSFT = MSFT.resample(rule='W').last()

#Compute the percentage change of prices
returns = MSFT.pct_change()

#Compute and print the autocorrelation of returns
autocorrelation = returns['Adj_Close'].autocorr()
print("The autocorrelation of weekly returns is %4.2f" %(autocorrelation))


#*****************************************************************************
# Interest Rates Autocorrelation
print('#*****************************************************************************\
     \n # Interest Rates Autocorrelation')
daily_data = read_fwf('daily_data', colspecs=[(0, 10), (12, 18), (19, 31)], parse_dates=['DATE'], index_col=['DATE'])
daily_data = daily_data.dropna()
print(daily_data['1962-01'])
# Compute the daily change in interest rates
daily_data['change_rates'] = daily_data.diff()

# Compute and print the autocorrelation of daily changes
autocorrelation_daily = daily_data['change_rates'].autocorr()
print("The autocorrelation of daily interest rate changes is %4.2f" %(autocorrelation_daily))

# Convert the daily data to annual data
annual_data = daily_data['US10Y'].resample('A').last()

# Repeat above for annual data
annual_data['diff_rates'] = annual_data.diff()
autocorrelation_annual = annual_data['diff_rates'].autocorr()
print("The autocorrelation of annual interest rate changes is %4.2f" %(autocorrelation_annual))


