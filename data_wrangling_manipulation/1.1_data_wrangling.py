import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# ****************************************************************
data = pd.read_csv('Customer data.csv')
print('**************Customer data analysis*****************')
# Selecting columns
unwanted_columns = ['Account Length', 'VMail Message', 'Day Calls']
columns = data.columns.values.tolist()
wanted_columns = [col for col in columns if col not in unwanted_columns]
subdata = data[wanted_columns]
print('Subset of data\n',subdata.head(),'\n\n')

# Selecting rows
print('Selecting first 9 rows: \n',subdata[1:10],'\n\n')  # [:10] is similar to [1:10]
# Selecting columns
print('Selecting only required columns: \n',subdata[['Intl Mins', 'Intl Calls', 'Intl Charge']].head(),'\n\n')

# Selecting a combination of rows and columns
sub_data1 = subdata['Intl Mins'][1:4]
subdata_first_50 = subdata[['Intl Mins', 'Intl Calls', 'Intl Charge']][1:50]

# # Using .loc[]  => using values
sub_data1 = subdata.loc[:, 'Eve Mins':'Eve Calls']                  # All rows, some columns
# sub_data1 = subdata.loc[1:10, :]                                  # Some rows, all columns
# sub_data1 = subdata.loc[1:10, 'Eve Mins':'Eve Calls']             # Some rows, some columns
# sub_data1 = subdata.loc[3:15, ['Eve Mins',  'Eve Calls']]         # Some rows, selected columns
# print(sub_data1)

# # Using .iloc[]  => using indices
sub_data1 = subdata.iloc[2:5, 1:5]
# sub_data1 = subdata.iloc[[0,4,5], 0:2]
# sub_data1 = subdata.iloc[1:10,4:7]
# sub_data1 = subdata.iloc[[1,2,5],[2,5,7]]
print('Subsetting only 2nd to 4th row and only selected columns:\n',sub_data1,'\n\n')



# Filter rows based on conditions
data1 = subdata.iloc[1:1001, :][(subdata['State'] == 'OH') & (subdata['Area Code'] == 415)]  # Both conditions
# data1 = subdata[(subdata['Night Mins'] > 200) | (subdata['State'] == 'VA')]       # Either condition
# print(subdata[subdata.State.isin(['OH', 'VA'])])
print('From first 1000 rows filter state = "OH" and Area code = "415"\n',data1,'\n\n')

# DataFrames with zeros
df2 = data1.copy()
df2['beta'] = [0, 0, 50, 60, 70, 80, 100, 150, 170]
df2['theta'] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(df2.any())            # df2.any() => True if even single value in columns is nonzeros
# print(df2.all())            # df2.all() => True if all value in columns with all nonzeros
df3 = df2.loc[:, df2.any()]
# print(df3)

# DataFrames with NaNs
# df3['kappa'] = [0, 0, None, 0, None, 0, 0, 0, 0]                         # Depricated method
# df3['iota'] = [None, None, None, None, None, None, None, None, None]     # Depricated method
df3 = df3.assign(kappa=[0, 0, None, 0, None, 0, 0, 0, 0])
df3 = df3.assign(iota=[None, None, None, None, None, None, None, None, None])
# print(df2.isnull().any())      # df2.isnull().any() => True if any value in columns is NaN
# print(df2.isnull().all())      # df2.isnull().all() => True if all values in columns is NaN
df4 = df3.loc[:, df3.notnull().any()]  # Remove columns that has all missing values
# print(df4)

# Creating new columns
df4['Total Mins'] = df4['Day Mins'] + df4['Eve Mins'] + df4['Night Mins']
print('Adding column "Total Mins=Day+Eve+Night Mins"\n',df4,'\n\n')
# Modifying a column based on another
df4.beta[df4['Intl Calls'] == 5] += 5
print('Added +5 to beta where "Intl calls" is 5 \n',df4)


# writing df4 to new csv file
df4.to_csv('newdf', index=False)


