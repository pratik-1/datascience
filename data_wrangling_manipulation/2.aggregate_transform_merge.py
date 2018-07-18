# Grouping the data – aggregation, filtering, and transformation

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# *****************
# Groupping

print('***************Grouping the Datasets**************')
# Generating dummy data ********
a=['Male','Female']
b=['Rich','Poor','Middle Class']
gender=[]
seb=[]
for i in range(1000):
    gender.append(np.random.choice(a))
    seb.append(np.random.choice(b))
height = 30*np.random.randn(1000) + 155          # mean(155) std.dev(30)
weight = 20*np.random.randn(1000) + 60           # mean(60) std.dev(20)
age = 5*np.random.randn(1000) + 35
income = 1500*np.random.randn(1000) + 15000


df = pd.DataFrame({'Gender':gender,
                   'Height': height,
                   'Weight': weight,
                   'Age': age,
                   'Income': income,
                   'Socio-Eco': seb})
# ************* 
# print(df)
grouped = df.groupby('Gender')
print('grouped by Gender: \n',grouped.groups,'\n\n')

# Get one from all groups
print('Get "Female" group from grouped data: \n',grouped.get_group('Female'),'\n\n')


# Get all groups
for names, group in grouped:
    print(names)
    # print(group)


# Multi-column groupping
grouped = df.groupby(['Gender', 'Socio-Eco'])
# Get Multi-column groupping
print('Group by Gender and Socio-Eco : \n',grouped.get_group(('Male', 'Rich')),'\n\n')

# # Get all groups
# for names, groups in grouped:
#     print(names, len(groups))
#     print(groups)


# Slice a column just like dataframe
grouped_income = grouped.get_group(('Male', 'Rich'))['Income']
print('"Income" of Male who are Rich :\n',grouped_income,'\n\n')


print('Sum of Groups by Gender and Socio-Eco : \n',grouped.sum(),'\n\n')    # get sum(similarly: max, mean, median)
print('Size of Groups by Gender and Socio-Eco : \n',grouped.size(),'\n\n')   # get number of records falling in the group
print('Count of Groups by Gender and Socio-Eco : \n',grouped.count(),'\n\n')  # get number of values in the group and columns
print('Description of Groups by Gender and Socio-Eco : \n',grouped.describe(),'\n\n\n\n')


# ****************************
# Aggregation
print('***************Aggregation the Datasets**************')
print('Group by Gender and Socio-Eco : ')
agg = grouped.aggregate({'Income': np.sum,'Age': np.mean,'Height': np.std})
agg.rename(columns={"Income": "Sum of Income", "Age":"mean of Age", "Height": "std in Height"}, inplace = True)
print(agg)
lam_agg = grouped.aggregate({'Age': np.mean, 'Height': lambda x: np.mean(x)/np.std(x)})
lam_agg.rename(columns={"Age":"mean of Age", "Height": "std in Height"}, inplace = True)
print(lam_agg)

# apply several functions to all the columns
all_in_one = grouped.aggregate([np.mean, np.std, np.size])   # Notice "()" is not given.
print(all_in_one)


# *******************************************
# Filtering
print('***************Filtering the Datasets **************')
# df2 = pd.DataFrame({'Weights': 15*np.random.randn(100)+80})
# print(df2.head())

# Get elements from Age column that are a part of the group wherein the sum of Age is greater than 700.
print('Group by Gender and Socio-Eco and elements from Age column that are a \
part of the group wherein the sum of Age is greater than 700 ')
print(grouped['Age'].filter(lambda x:x.sum()>700))


# ***********************************
# Transformation

# df['Income'] = df.Income.transform(lambda x: x/1000)
# print(df.head())

# calculate the standard normal values for all the elements in the numerical columns
# Z-Score
df2 = df.loc[:, df.dtypes == 'float64']
zscore = lambda x: (x - x.mean()) / x.std()
z_trans = df2.transform(zscore)
print('\n\nZ Scores top records\n',z_trans.head())

# Fillers
# filler = lambda x: x.fillna(x.mean())
# filled = grouped.transform(filler)
# print(filled)

#print(df.head())
df1 = df.sort_values(['Age','Income'])
grouped = df1.groupby('Gender')
print('\n\nSorted by Age and Income and grouped by Gender and fetch first record\n',grouped.head(1))






















































