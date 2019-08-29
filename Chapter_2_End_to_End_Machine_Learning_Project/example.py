import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

def load_housing_data(housing_path="dataset\housing"):
    csv_path= os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def split_train_test(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size] #Everything but the test_set_size last elements
    train_indices=shuffled_indices[test_set_size:] #Everything until the test_set_size elements
    return data.iloc[train_indices], data.iloc[test_indices]

housing=load_housing_data()

'''
print(housing)
housing.info() #Or print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())
#Show the histograms
housing.hist(bins=50,figsize=(20,15))
plt.show()
#print(housing.describe())
'''

'''
train_set, test_set= split_train_test(housing,0.2)
print(len(train_set),"train +", len(test_set),"test")
'''

'''
housing_with_id=housing.reset_index()
train_set, test_set=split_train_test_by_id(housing_with_id,0.2,"index")
print(len(train_set),"train +", len(test_set),"test")
'''

'''
train_set,test_set=train_test_split(housing, test_size=0.2, random_state=42)
'''


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
#housing.hist(bins=50,figsize=(20,15),column='income_cat')
#plt.show()



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
#print(housing["income_cat"].value_counts() / len(housing))
#housing = strat_train_set.copy()

#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
'''
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population",
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
#plt.show()
'''

'''
corr_matrix= housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
'''

attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
'''
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)
plt.show()
'''
corr_matrix=housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
