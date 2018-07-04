# Splitting a dataset in training and testing datasets

# Method 1 - Random sampling
import pandas as pd
import numpy as np

filepath = '/media/pratik/volume_D/courses/Datacamp/datasets/Datasets_for_Predictive_Modelling_Python\
/Chapter 3/'

# data = pd.read_csv(filepath+'Customer Churn Model.txt')
# n_rec = len(data)
# print(n_rec)
# a = np.random.random(n_rec)
# check = a < 0.8
# train = data[check]
# test = data[~check]


# # Method 2 – using sklearn
# from sklearn.model_selection import train_test_split
#
# train, test = train_test_split(data, test_size=0.2)
# print(len(train))
# print(len(test))



# Method 3 – using the shuffle function
record = []
# with open(filepath+'Customer Churn Model.txt', 'r') as f:
#     data = f.read().split('\n')
#
#     for l in data:
#         print(l)
#         record.append(data)
# np.random.shuffle(data)
# check = int(0.8*len(data))
# train_data = data[:check]
# test_data = data[check:]
# print(len(train_data))
# print(len(test_data))
# print(record)


# **********************************************************************
# Concatenating


data1 = pd.read_csv(filepath+'winequality-red.csv', sep=';')
data2 = pd.read_csv(filepath+'winequality-white.csv', sep=';')

print(data1.shape)
print(data2.shape)
tot = pd.concat([data1, data2], axis=0)
print(tot.shape)


# ***************
path= filepath + '/lotofdata/'

data3 = pd.read_csv(path+'001.csv')
print(data3.shape)
data = []
# for i in range(1, 333):
#     file = path+format(i, "03d")+'.csv'
#     df = pd.read_csv(file)
#     data.append(df)
# # print(data)
# combined = pd.concat([d for d in data], axis=0, ignore_index=True)
#
# # combined.to_csv(filepath+'combined', index=False)
#
# print(combined.info())

# ***************************************************************
# Merge

path = filepath + 'Medals/'
print(path)
medals = pd.read_csv(path + 'medals.csv', parse_dates=True, encoding='utf-8')
print('Medals')
print(medals.info())
modified_medals = medals[medals.Athlete.notnull()]
print('modified_medals')
print(modified_medals.info())

ath_country_map = pd.read_csv(path + 'Athelete_Country_Map.csv')
ath_sports_map = pd.read_csv(path + 'Athelete_Sports_Map.csv')

# Removed Null in Athlete, dropped duplicate Athlete in countrymap
modified_country_map = ath_country_map[ath_country_map.Athlete.notnull()]
print('modified_country_map')
print(modified_country_map.info())
deduped_modified_country_map = modified_country_map.drop_duplicates(subset='Athlete')
print('deduped_modified_country_map')
print(deduped_modified_country_map.info())

# Merge medals with country map
med_ath_cntry = pd.merge(modified_medals, deduped_modified_country_map, left_on='Athlete', right_on='Athlete')
print('med_ath_cntry')
print(med_ath_cntry.info())
# Remove some of the athletes
rem_aths = ['Michael Phelps', 'Natalie Coughlin', 'Chen Jing', 'Richard Thompson', 'Matt Ryan']
country_map_dlt=deduped_modified_country_map[~deduped_modified_country_map.Athlete.isin(rem_aths)]
print('country_map_dlt')
print(country_map_dlt.info())


print('Athelete_Sports_Map')
print(ath_sports_map.info())
deduped_ath_sports_map = ath_sports_map.drop_duplicates(subset='Athlete').dropna()
print('deduped_ath_sports_map')
print(deduped_ath_sports_map.info())


med_sports = pd.merge(modified_medals, deduped_ath_sports_map, left_on='Athlete', right_on='Athlete')
print('med_sports')
print(med_sports.info())

rem_aths = ['Michael Phelps', 'Natalie Coughlin', 'Chen Jing', 'Richard Thompson', 'Matt Ryan']
ath_sports_map_dlt = deduped_ath_sports_map[~deduped_ath_sports_map.Athlete.isin(rem_aths)]
print('sports_map_dlt')
print(ath_sports_map_dlt.info())




# Inner	Join
merged_inner = pd.merge(left=med_ath_cntry, right=country_map_dlt, how='inner', left_on='Athlete', right_on='Athlete')
print('merged_inner :', len(merged_inner))
print(merged_inner[merged_inner.isnull().any(axis=1)])

# Left Join
merged_left = pd.merge(left=med_ath_cntry, right=country_map_dlt, how='left', left_on='Athlete', right_on='Athlete')
print('merged_left :', len(merged_left))
# print(merged_left[merged_left.isnull().any(axis=1)])

# Right Join
merged_right = pd.merge(left=med_ath_cntry, right=country_map_dlt, how='right', left_on='Athlete', right_on='Athlete')
print('merged_right :', len(merged_right))
# print(merged_right[merged_right.isnull().any(axis=1)])

# Outer Join
merged_outer = pd.merge(left=med_ath_cntry, right=country_map_dlt, how='outer', left_on='Athlete', right_on='Athlete')
print('merged_outer :', len(merged_outer))
# print(merged_outer[merged_outer.isnull().any(axis=1)])

