
# Chapter 2: End to End Machine Learning Project
In this chapter we will create a full project, first we will see the main steps we will be doing:
1. [Look at the big picture](#look-at-the-big-picture)
2. [Get the data](#get-the-data)
3. [Discover and visualize the data to gain insights](#discover-and-visualize-the-data-to-gain-insights)
4. [Prepare the data for Machine Learning algorithms](#prepare-the-data-for-machine-learning-algorithms)
5. [Select a model and train it](#select-a-model-and-train-it)
6. [Fine-tune your model](#fine-tune-your-model)
7. [Launch, monitor, and maintain your system](#launch-monitor-and-maintain-your-system)

## Data we will be using
We will be using the California Housing Prices dataset from the StatLib repository. This dataset was based on data from the 1990 California census. It is not exactly recent, but it has many qualities for learning, so we will pretend it is recent data. We also added a categorical attribute and removed a few features for teaching purposes.
## Look at the big picture
The first task you are asked to perform is to build a model of housing prices in California using the California census data.

Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.
### Frame the Problem
The first question to ask is what exactly is the business objective; building a model is probably not the end goal. This is important because it will determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it.

Our model’s output (a prediction of a district’s median housing price) will be fed to another Machine Learning system, along with many other signals. This downstream system will determine whether it is worth investing in a given area or not. Getting this right is critical, as it directly affects revenue.

![Pipeline](../img/chp2_pipeline.png?raw=true "Pipeline")

#### Pipelines
A sequence of data processing components is called a data pipeline. Pipelines are very common in Machine Learning systems, since there is a lot of data to manipulate and many data transformations to apply.

First, you need to frame the problem: is it **supervised**, **unsupervised**, or **Reinforcement Learning**? Is it a **classification** task, a **regression** task, or something else? Should you use **batch learning** or **online learning** techniques?

You can check out this terms in the [previous chapter](Chapter_1_The_machine_Learning_Landscape).

Let’s see: it is clearly a typical **supervised learning** task since you are given labeled training examples (each instance comes with the expected output, i.e., the district’s median housing price).
 Moreover, it is also a typical **regression** task, since you are asked to predict a value. More specifically, this is a **multivariate regression** problem since the system will use multiple features to make a prediction (it will use the district’s population, the median income, etc.).

Finally, there is no continuous flow of data coming in the system, there is no particular need to adjust to changing data rapidly, and the data is small enough to fit in memory, so **plain batch learning** should do just fine.
### Select a performence measure

The next step is to select a performance measure. A typical performance measure for regression problems is the Root Mean Square Error (RMSE). It measures the standard deviation of the errors the system makes in its predictions.
*For example*, an RMSE equal to 50,000 means that about 68% of the system’s predictions fall within $50,000 of the actual value, and about 95% of the predictions fall within $100,000 of the actual value.


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?RMSE%28X%2Ch%29%3D%5Csqrt%7B%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%20%28h%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29%5E2%7D" alt="RMSE Equation"/>
 </p>

#### Notations
This equation introduces several very common Machine Learning notations.
- **m** is the number of instances in the dataset you are measuring the RMSE on.
	- *For example*, if you are evaluating the RMSE on a validation set of 2,000 districts,
then m = 2,000.
- **x^(i)^** is a vector of all the **feature values** (excluding the label) of the ith instance in
the dataset, and **y^(i)^** is its **label** (the desired output value for that instance).
	- *For example*, if the first district in the dataset is located at longitude –118.29°,
latitude 33.91°, and it has 1,416 inhabitants with a median income of $38,372,
and the median house value is $156,400 (ignoring the other features for now),
then:
<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?x%5E%7B%281%29%7D%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D-118.29%20%5C%5C%2033.91%20%5C%5C1.416%20%5C%5C38.372%20%5Cend%7Barray%7D%5Cright%29" alt="Equation"/>
</p>
  And:
<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?y%5E%7B%281%29%7D%3D156.400" alt="Equation"/>
</p>

- **X** is a matrix containing all the feature values (excluding labels) of all instances in the dataset. There is one row per instance and the *i^th^* row is equal to the transpose of *x^(i)^*, noted *(x^(i)^)^T^*.
	- *For example*, if the first district is as just described, then the matrix X looks
like this:

<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%7B%28x%5E%7B%281%29%7D%29%5ET%7D%5C%5C%20%7B%28x%5E%7B%282%29%7D%29%5ET%7D%20%5C%5C%20%3A%20%5C%5C%20%7B%28x%5E%7B%281999%29%7D%29%5ET%7D%20%5C%5C%20%7B%28x%5E%7B%282000%29%7D%29%5ET%7D%5Cend%7Barray%7D%5Cright%29%20%3D%20%5Cbegin%7Bpmatrix%7D%20-118.29%20%26%2033.91%20%26%201.%20416%20%26%2038.372%5C%5C%20%3A%20%26%20%3A%20%26%20%3A%20%26%20%3A%20%5Cend%7Bpmatrix%7D" alt="Equation"/>
</p>

- ***h*** is your system’s prediction function, also called a hypothesis. When your system is given an instance’s feature vector x^(i)^, it outputs a predicted value ŷ^(i)^ = *h*(x^(i)^) for that instance (ŷ is pronounced “y-hat”).
	- *For example*, if your system predicts that the median housing price in the first district is $158,400, then ŷ^(1)^ = *h*(x^(1)^) = 158,400. The **prediction error** for this district is ŷ^(1)^ – y^(1)^ = 158,400 - 156.400 = 2,000.
- RMSE(**X**,*h*) is the cost function measured on the set of examples using your
hypothesis *h*.

We use lowercase italic font for scalar values (such as *m* or *y^(i)^*) and function names
(such as *h*), lowercase bold font for vectors (such as **x**^(i)^), and uppercase bold font for
matrices (such as **X**).

Even though the RMSE is generally the preferred performance measure for regression
tasks, in some contexts you may prefer to use another function. For example, suppose
that there are many outlier districts. In that case, you may consider using the *Mean
Absolute Error*

<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?MAE%28X%2Ch%29%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%5Cleft%20%7C%20h%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%20%5Cright%7C" alt="Equation"/>
</p>

Both the RMSE and the MAE are ways to **measure the distance between two vectors**:
the vector of predictions and the vector of target values. Various distance measures,
or *norms*, are possible:

- Computing the root of a sum of squares (RMSE) corresponds to the *Euclidian norm*: it is the notion of distance you are familiar with. It is also called the ℓ<sub>2</sub> *norm*, noted || · ||<sub>2</sub> (or just || · ||).
- Computing the sum of absolutes (MAE) corresponds to the ℓ<sub>1</sub> *norm*, noted || · ||<sub>1</sub>. It is sometimes called the *Manhattan norm* because it measures the distance between two points in a city if you can only travel along orthogonal city blocks.
 - More generally, the ℓ~k~ *norm* of a vector **v** containing *n* elements is defined as
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20v%20%5Cright%20%5C%7C_k%3D%28%5Cleft%20%7C%20v_0%20%5Cright%20%7C%5Ek%20&plus;%20%5Cleft%20%7C%20v_1%20%5Cright%20%7C%5Ek%20&plus;%20%5Ccdots%20&plus;%5Cleft%20%7C%20v_n%20%5Cright%20%7C%5Ek%29%5E%7B%5Cfrac%7B1%7D%7Bk%7D%7D" alt="Equation"/>
</p>

  ℓ<sub>0</sub> just gives the cardinality of the vector (i.e., the number of elements), and ℓ<sub>∞</sub> gives the maximum absolute value in the vector.
 - The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.


## Get the data
The data we will be using can be found [here](https://github.com/ageron/handson-ml/tree/master/datasets/housing).
### Create the Workspace
Make sure you have everything installed as we saw in the [Installation](https://github.com/xamo1998/Machine-Learning#installation) guide.
### Quick look of the data structure
Once you have the CSV file we are going to create a python code that read all the data.

We will be using the module *pandas*:
```python
import pandas as pd
import os

def load_housing_data(housing_path="dataset\housing"):
    csv_path= os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()
print(housing)
```
In order of this to work you need to have you *.csv* file inside the following directory:
*/(directory_of_.py)/dataset/housing/housing.csv*

If your data is in other path you can change de code.

If we run the program, we can see that the *.csv* file has been readed correctly:

![Housing Data](../img/chp2_housing_data_printed.png?raw=true "Housing Data")

Each row represents one district. There are 10 attributes (longitude, latitude...)

The *info()* method is useful to get a quick description of the data we are using, in particular the total number of rows and each attribute's type and number of *non-null* values.

```python
...
...
housing=load_housing_data()
housing.info() #Or print(housing.info())
```

If we run the code below we will get the following output:

![Housing Data Info](../img/chp2_housing_data_info.PNG?raw=true "Housing Data Info")

There are 20,640 instances in the dataset, which means that is fairly small by Machine Learning standards, but is perfect to get started.

Notice that the *total_bedrooms* attribute has only 20,433 *non-null* values, meaning that 207 districts are missing this feature. We will need to take care of this later.

All attributes are numerical, except the *ocean_proximity* field. Its type is object, so it could hold any kind of Python object, but since you loaded this data from a CSV file you know that it must be a text attribute. When you looked at the top five rows, you probably noticed that the values in that column were repetitive, which means that it is probably a categorical attribute. You can find out what categories exist and how many districts belong to each category by using the *value_counts()* method:

```python
...
...
housing=load_housing_data()
print(housing['ocean_proximity'].value_counts())
```

If we run the code below we will get the following output:

![Housing Data Value Counts](../img/chp2_housing_data_value_counts.PNG?raw=true "Housing Data Value Counts")

Let's look at the other fields. The *desribe()* method shows a summary of the numerical attributes:

```python
...
...
housing=load_housing_data()
print(housing.describe())
```

If we run the code below we will get the following output:

![Housing Data Describe](../img/chp2_housing_data_describe.PNG?raw=true "Housing Data Describe")

The *count*, *mean*, *min*, and *max* rows are self-explanatory. Note that the *null values* are ignored (so, for example, count of *total_bedrooms* is 20,433, not 20,640). The *std* row shows the standard deviation (which measures how dispersed the values are).

The 25%, 50%, and 75% rows show the corresponding percentiles: **A percentile indicates the value below which a given percentage of observations in a group of observations falls**. For example, 25% of the districts have a *housing_median_age* lower than 18, while 50% are lower than 29 and 75% are lower than 37. These are often called the 25th percentile (or 1st quartile), the median, and the 75th percentile (or 3rd quartile).

Another quick way to get a feel of the type of data you are dealing with is to plot a histogram for each numerical attribute. **A histogram shows the number of instances (on the vertical axis) that have a given value range (on the horizontal axis)**. You can either plot this one attribute at a time, or you can call the *hist()* method on the whole dataset, and it will plot a histogram for each numerical attribute. For example, you can see that slightly over 800 districts have a *median_house_value* equal to about $500,000.

```python
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_housing_data(housing_path="dataset\housing"):
    csv_path= os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()
housing.hist(bins=50,figsize=(20,15))
plt.show()
```

If we run the code below we will get the following output:

![Housing Data Plot](../img/chp2_housing_data_plot.png?raw=true "Housing Data Plot")

Notice a few things in these histograms:

1. First, the median income attribute does not look like it is expressed in US dollars (USD). After checking with the team that collected the data, you are told that the data has been scaled and capped at 15 (actually 15.0001) for higher median incomes, and at 0.5 (actually 0.4999) for lower median incomes. Working with preprocessed attributes is common in Machine Learning, and it is not necessarily a problem, but you should try to understand how the data was computed.

2. The housing median age and the median house value were also capped. The latter may be a serious problem since it is your target attribute (your labels). Your Machine Learning algorithms may learn that prices never go beyond that limit. You need to check with your client team (the team that will use your system’s output) to see if this is a problem or not. If they tell you that they need precise predictions even beyond $500,000, then you have mainly two options:
    - Collect proper labels for the districts whose labels were capped.
    - Remove those districts from the training set (and also from the test set, since your system should not be evaluated poorly if it predicts values beyond $500,000).

3. These attributes have very different scales. We will discuss this later in this chapter when we explore feature scaling.

4. Finally, many histograms are *tail heavy*: they extend much farther to the right of the median than to the left. This may make it a bit harder for some Machine Learning algorithms to detect patterns. We will try transforming these attributes later on to have more bell-shaped distributions.

Hopefully you now have a better understanding of the kind of data you are dealing with.
### Create a Test set
It may sound strange to voluntarily set aside part of the data at this stage. After all,
you have only taken a quick glance at the data, and surely you should learn a whole
lot more about it before you decide what algorithms to use, right? This is true, but
your brain is an amazing pattern detection system, which means that it is highly
prone to overfitting: if you look at the test set, you may stumble upon some seemingly
interesting pattern in the test data that leads you to select a particular kind of
Machine Learning model. When you estimate the generalization error using the test
set, your estimate will be too optimistic and you will launch a system that will not
perform as well as expected. This is called *data snooping* bias.

Creating a test set is theoretically quite simple: just pick some instances randomly,
typically 20% of the dataset, and set them aside:
```python
...
import numpy as np
...
def split_train_test(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size] #Everything but the test_set_size last elements
    train_indices=shuffled_indices[test_set_size:] #Everything until the test_set_size elements
    return data.iloc[train_indices], data.iloc[test_indices]

housing=load_housing_data()
...
train_set, test_set= split_train_test(housing,0.2)
print(len(train_set),"train +", len(test_set),"test")
```

If we run the code below we will get the following output:

![Housing Data Splitted](../img/chp2_housing_data_splitted.png?raw=true "Housing Data
Splitted")

Well, this works, but it is not perfect: if you run the program again, it will generate a different test set! Over time, you (or your Machine Learning algorithms) will get to see the whole dataset, which is what you want to avoid.

One solution is to save the test set on the first run and then load it in subsequent runs. Another option is to set the random number generator’s seed (e.g., *np.ran dom.seed(42))* before calling *np.random.permutation()*, so that it always generates the same shuffled indices.

But both these solutions will break next time you fetch an updated dataset. A common solution is to use each instance’s identifier to decide whether or not it should go in the test set (assuming instances have a unique and immutable identifier).

For example, you could compute a hash of each instance’s identifier, keep only the last byte of the hash, and put the instance in the test set if this value is lower or equal to 51 (~20% of 256). This ensures that the test set will remain consistent across multiple runs, even if you refresh the dataset. The new test set will contain 20% of the new instances, but it will not contain any instance that was previously in the training set.

Here is a possible implementation:

```python
...
import hashlib
...
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

Unfortunately, the housing dataset does not have an identifier column. The simplest solution is to use the row index as the ID:

```python
housing_with_id=housing.reset_index()
train_set, test_set=split_train_test_by_id(housing_with_id,0.2,"index")
```

If you use the row index as a unique identifier, you need to make sure that new data gets appended to the end of the dataset, and no row ever gets deleted. If this is not possible, then you can try to use the most stable features to build a unique identifier.

For example, a district’s latitude and longitude are guaranteed to be stable for a few
million years, so you could combine them into an ID like so:

```python
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
```

Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways. The simplest function is *train_test_split*, which does pretty much the same thing as the function *split_train_test* defined earlier, with a couple of additional features. First there is a random_state parameter that allows you to set the random generator seed as explained previously, and second you can pass it multiple datasets with an identical number of rows, and it will split them on the same indices (this is very useful, for example, if you have a separate DataFrame for labels):

```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

So far we have considered purely random sampling methods. This is generally fine if your dataset is large enough (especially relative to the number of attributes), but if it is not, you run the risk of introducing a significant sampling bias. When a survey company decides to call 1,000 people to ask them a few questions, they don’t just pick 1,000 people randomly in a phone booth. They try to ensure that these 1,000 people are representative of the whole population. For example, the US population is composed of 51.3% female and 48.7% male, so a well-conducted survey in the US would try to maintain this ratio in the sample: 513 female and 487 male. This is called *stratified sampling*: the population is divided into homogeneous subgroups called *strata*, and the right number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population. If they used purely random sampling, there would be about 12% chance of sampling a skewed test set with either less than 49% female or more than 54% female. Either way, the survey results would be significantly biased.

Suppose you chatted with experts who told you that the median income is a very important attribute to predict median housing prices. You may want to ensure that the test set is representative of the various categories of incomes in the whole dataset. Since the median income is a continuous numerical attribute, you first need to create an income category attribute. Let’s look at the median income histogram more closely:

![Housing Median Income Plot](../img/chp2_housing_median_income_plot.png?raw=true "Housing Median Income Plot")

Most median income values are clustered around 2–5 (tens of thousands of dollars), but some median incomes go far beyond 6. It is important to have a sufficient numberof instances in your dataset for each stratum, or else the estimate of the stratum’s importance may be biased. This means that you should not have too many strata, and each stratum should be large enough. The following code creates an income category attribute by dividing the median income by 1.5 (to limit the number of income categories), and rounding up using ceil (to have discrete categories), and then merging all the categories greater than 5 into category 5:

```python
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
```

After this the histogram will looks like:

![Housing Income Cat Plot](../img/chp2_housing_income_cat_plot.png?raw=true "Housing Income Cat Plot")

Now you are ready to do stratified sampling based on the income category. For this
you can use Scikit-Learn’s *StratifiedShuffleSplit* class:
```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]
```
Let’s see if this worked as expected. You can start by looking at the income category
proportions in the full housing dataset:
```python
print(housing["income_cat"].value_counts() / len(housing))
```

![Housing Data Stratified](../img/chp2_housing_data_stratified.PNG?raw=true "Housing Data Stratified")


Now you should remove the income_cat attribute so the data is back to its original
state:
```python
for set in (strat_train_set, strat_test_set):
  set.drop(["income_cat"], axis=1, inplace=True)
```
We spent quite a bit of time on test set generation for a good reason: this is an often neglected but critical part of a Machine Learning project. Moreover, many of these ideas will be useful later when we discuss cross-validation. Now it’s time to move on to the next stage: exploring the data.
## Discover and visualize the data to gain insights
So far you have only taken a quick glance at the data to get a general understanding of the kind of data you are manipulating. Now the goal is to go a little bit more in depth.

First, make sure you have put the test set aside and you are only exploring the training set. Also, if the training set is very large, you may want to sample an exploration set, to make manipulations easy and fast. In our case, the set is quite small so you can just work directly on the full set. Let’s create a copy so you can play with it without harming the training set:
```python
housing = strat_train_set.copy()
```

### Visualizing Geographical DataFrame
Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data:
```python
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
```

If we run the code below we will get the following output:

![Housing Geographical Data](../img/chp2_housing_geographical_data.png?raw=true "Housing Geographical Data")

This looks like California all right, but other than that it is hard to see any particular pattern. Setting the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points:

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
```
If we run the code below we will get the following output:

![Housing Geographical Data](../img/chp2_housing_geographical_data_alpha01.png?raw=true "Housing Geographical Data")

Now that’s much better: you can clearly see the high-density areas, namely the Bay Area and around Los Angeles and San Diego, plus a long line of fairly high density in the Central Valley, in particular around Sacramento and Fresno.

More generally, our brains are very good at spotting patterns on pictures, but you may need to play around with visualization parameters to make the patterns stand out.

Now let’s look at the housing prices. The radius of each circle represents the district’s population (option s), and the color represents the price (option c). We will use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices):
```Python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population",
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
plt.show()
```

The output is the following:

![Housing Geographical Data Full](../img/chp2_housing_geographical_data_full.png?raw=true "Housing Geographical Data Full")

This image tells us that the housing prices are very much related to the location (e.g., close to the ocean) and to the population density, as you probably knew already. It will probably be useful to use a clustering algorithm to detect the main clusters, and add new features that measure the proximity to the cluster centers. The ocean proximity attribute may be useful as well, although in Northern California the housing
prices in coastal districts are not too high, so it is not a simple rule.
### Looking for Correlations
Since the dataset is not too large, you can easily compute the *standard correlation coefficient* (also called *Pearson’s r*) between every pair of attributes using the *corr()* method:
```Python
corr_matrix = housing.corr()
```

Now let’s look at how much each attribute correlates with the median house value:
```Python
print(corr_matrix["median_house_value"].sort_values(ascending=False))
```

The output is the following:

![Housing Correlation](../img/chp2_housing_corr.png?raw=true "Housing Correlation")

The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value (i.e., prices have a slight tendency to go down when you go north). Finally, coefficients close to zero mean that there is no linear correlation.

Another way to check for correlation between attributes is to use Pandas’ *scatter_matrix* function, which plots every numerical attribute against every other numerical attribute. Since there are now 11 numerical attributes, you would get 11^2 = 121 plots, which would not fit on a page, so let’s just focus on a few promising attributes that seem most correlated with the median housing value:

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
```

The output is the following:

![Housing Correlation All](../img/chp2_housing_corr_all.png?raw=true "Housing Correlation All")

The main diagonal (top left to bottom right) would be full of straight lines if Pandas plotted each variable against itself, which would not be very useful. So instead Pandas displays a histogram of each attribute (other options are available; see Pandas’ documentation for more details).

The most promising attribute to predict the median house value is the median income, so let’s zoom in on their correlation scatterplot:
```python
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
```
The output is the following:

![Housing Correlation Median Income](../img/chp2_housing_corr_median_income.png?raw=true "Housing Correlation Median Income")


This plot reveals a few things. First, the correlation is indeed very strong; you can clearly see the upward trend and the points are not too dispersed. Second, the price cap that we noticed earlier is clearly visible as a horizontal line at $500,000. But this plot reveals other less obvious straight lines: a horizontal line around $450,000, another around $350,000, perhaps one around $280,000, and a few more below that. You may want to try removing the corresponding districts to prevent your algorithms from learning to reproduce these data quirks.

### Experimenting with Attribute Combinations
Hopefully the previous sections gave you an idea of a few ways you can explore the data and gain insights. You identified a few data quirks that you may want to clean up before feeding the data to a Machine Learning algorithm, and you found interesting correlations between attributes, in particular with the target attribute. You also noticed that some attributes have a tail-heavy distribution, so you may want to transform them (e.g., by computing their logarithm). Of course, your mileage will vary considerably with each project, but the general ideas are similar.

One last thing you may want to do before actually preparing the data for Machine Learning algorithms is to try out various attribute combinations. For example, the total number of rooms in a district is not very useful if you don’t know how many households there are. What you really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful: you probably want to compare it to the number of rooms. And the population per household also seems like an interesting attribute combination to look at. Let’s create these new attributes:
```Python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```

And now let's look at the correlation matrix again:
```Python
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
```

The output is the following:

![Housing Correlation Combining Attributes](../img/chp2_combining.png?raw=true "Housing Correlation Combining Attributes")

Hey, not bad! The new *bedrooms_per_room* attribute is much more correlated with the median house value than the total number of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio tend to be more expensive. The number of rooms per household is also more informative than the total number of rooms in a district—obviously the larger the houses, the more expensive they are.

This round of exploration does not have to be absolutely thorough; the point is to start off on the right foot and quickly gain insights that will help you get a first reasonably good prototype. But this is an iterative process: once you get a prototype up and running, you can analyze its output to gain more insights and come back to this exploration step.
## Prepare the data for Machine Learning algorithms
It’s time to prepare the data for your Machine Learning algorithms. Instead of just doing this manually, you should write functions to do that, for several good reasons:
- This will allow you to reproduce these transformations easily on any dataset (e.g., the next time you get a fresh dataset).

- You will gradually build a library of transformation functions that you can reuse in future projects.

- You can use these functions in your live system to transform the new data before feeding it to your algorithms.
- This will make it possible for you to easily try various transformations and see which combination of transformations works best.

But first let’s revert to a clean training set (by copying *strat_train_set* once again), and let’s separate the predictors and the labels since we don’t necessarily want to apply the same transformations to the predictors and the target values (note that *drop()* creates a copy of the data and does not affect *strat_train_set*):
```Python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```
### Data Cleaning
Most Machine Learning algorithms cannot work with missing features, so let’s create a few functions to take care of them. You noticed earlier that the total_bedrooms attribute has some missing values, so let’s fix this. You have three options:
- Get rid of the corresponding districts.

- Get rid of the whole attribute.

- Set the values to some value (zero, the mean, the median, etc.).

You can accomplish these easily using DataFrame’s *dropna()*, *drop()*, and *fillna()* methods:
```Python
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median) # option 3
```
If you choose option 3, you should compute the median value on the training set, and use it to fill the missing values in the training set, but also don’t forget to save the median value that you have computed. You will need it later to replace missing values in the test set when you want to evaluate your system, and also once the system goes live to replace missing values in new data.

Scikit-Learn provides a handy class to take care of missing values: *Imputer*. Here is how to use it. First, you need to create an Imputer instance, specifying that you want to replace each attribute’s missing values with the median of that attribute:
```Python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```
Since the median can only be computed on numerical attributes, we need to create a copy of the data without the text attribute ocean_proximity:
```Python
housing_num = housing.drop("ocean_proximity", axis=1)
```

Now you can fit the imputer instance to the training data using the *fit()* method:
```Python
imputer.fit(housing_num)
```
The imputer has simply computed the median of each attribute and stored the result in its *statistics_ instance* variable. Only the *total_bedrooms* attribute had missing values, but we cannot be sure that there won’t be any missing values in new data after the system goes live, so it is safer to apply the imputer to all the numerical attributes:
```Python
print(imputer.statistics_)
```
The ouput is the following:

![Imputer Stats](../img/chp2_imputer_stats.PNG?raw=true "Imputer Stats")

**Now you can use this “trained” imputer to transform the training set by replacing missing values by the learned medians:**
```Python
X = imputer.transform(housing_num)
```
The result is a plain *Numpy* array containing the transformed features. If you want to put it back into a Pandas *DataFrame*, it’s simple:
```Python
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
```
>#### Scikit-Learn Desgin
  Scikit-Learn Design
  Scikit-Learn’s API is remarkably well designed. The main design principles are:
  - **Consistency**. All objects share a consistent and simple interface:
    - *Estimators*. Any object that can estimate some parameters based on a dataset is called an *estimator* (e.g., an imputer is an estimator). The estimation itself is performed by the *fit()* method, and it takes only a dataset as a parameter (or two for supervised learning algorithms; the second dataset contains the labels). Any other parameter needed to guide the estimation process is considered a hyperparameter (such as an imputer’s strategy), and it must be set as an instance variable (generally via a constructor parameter).
    - *Transformers*. Some estimators (such as an imputer) can also transform a dataset; these are called *transformers*. Once again, the API is quite simple: the transformation is performed by the *transform()* method with the dataset to transform as a parameter. It returns the transformed dataset. This transformation generally relies on the learned parameters, as is the case for an imputer. All transformers also have a convenience method called *fit_transform()* that is equivalent to calling *fit()* and then *transform()* (but sometimes *fit_transform()* is optimized and runs much faster).
    - *Predictors*. Finally, some estimators are capable of making predictions given a dataset; they are called *predictors*. For example, the LinearRegression model in the previous chapter was a predictor: it predicted life satisfaction given a country’s GDP per capita. A predictor has a *predict()* method that takes a dataset of new instances and returns a dataset of corresponding predictions. It also has a *score()* method that measures the quality of the predictions given a test set (and the corresponding labels in the case of supervised learning algorithms).
  - **Inspection**. All the estimator’s hyperparameters are accessible directly via public instance variables (e.g., *imputer.strategy*), and all the estimator’s learned parameters are also accessible via public instance variables with an underscore suffix (e.g., *imputer.statistics_*).
  - **Nonproliferation of classes**. Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers.
  - **Composition**. Existing building blocks are reused as much as possible. For example, it is easy to create a Pipeline estimator from an arbitrary sequence of transformers followed by a final estimator, as we will see.
  - **Sensible defaults**. Scikit-Learn provides reasonable default values for most parameters, making it easy to create a baseline working system quickly.

### Handling Text and Categorical attributes
Earlier we left out the categorical attribute ocean_proximity because it is a text attribute so we cannot compute its median. Most Machine Learning algorithms prefer to work with numbers anyway, so let’s convert these text labels to numbers.

Scikit-Learn provides a transformer for this task called *LabelEncoder*:
```Python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
```
Output:

![Label Encoder](../img/chp2_label_encoder.png?raw=true "Label Encoder")

Now we can use this numerical data in any ML algorithm. You can look at the mapping that this encoder has learned using the *classes_* attribute (“<1H OCEAN” is mapped to 0, “INLAND” is mapped to 1, etc.):
```Python
print(encoder.classes_)
```
Output:

![Classes Decoded](../img/chp2_decode_classes.png?raw=true "Classes Decoded")

One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values. Obviously this is not the case (for example, categories 0 (<1H OCEAN) and 4 (NEAR OCEAN) are more similar than categories 0 (<1H OCEAN) and 1 (INLAND)). To fix this issue, a common solution is to **create one binary attribute per category**: one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is “INLAND” (and 0 otherwise), and so on. This is called *one-hot encoding*, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold).

Scikit-Learn provides a *OneHotEncoder* encoder to convert integer categorical values into one-hot vectors. Let’s encode the categories as one-hot vectors. Note that *fit_transform()* expects a 2D array, but *housing_cat_encoded* is a 1D array, so we need to reshape it:
```Python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot)
```
Output:

![One Hot Encoder](../img/chp2_one_hot_encoder.png?raw=true "One Hot Encoder")

Notice that the output is a SciPy sparse matrix, instead of a NumPy array. This is very useful when you have categorical attributes with thousands of categories. After onehot encoding we get a matrix with thousands of columns, and the matrix is full of zeros except for one 1 per row. Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the nonzero elements. You can use it mostly like a normal 2D array,19 but if you really want to convert it to a (dense) NumPy array, just call the *toarray()* method:
```Python
print(housing_cat_1hot.toarray())
```
Output:

![One Hot Encoder Array](../img/chp2_one_hot_encoder_array.png?raw=true "One Hot Encoder Array")

We can apply both transformations (from text categories to integer categories, then from integer categories to one-hot vectors) in one shot using the *LabelBinarizer* class:
```Python
from sklearn.preprocessing import LabelBinarizer

housing_cat = housing["ocean_proximity"]
encoder= LabelBinarizer()
housing_cat_1hot=encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
```
Output:

![One Hot Encoder Array](../img/chp2_one_hot_encoder_array.png?raw=true "One Hot Encoder Array")

Note that this returns a dense NumPy array by default. You can get a sparse matrix instead by passing *sparse_output=True* to the *LabelBinarizer* constructor.

### Custom Transformers
Although Scikit-Learn provides many useful transformers, you will need to write your own for tasks such as custom cleanup operations or combining specific attributes. You will want your transformer to work seamlessly with Scikit-Learn functionalities (such as pipelines), and since Scikit-Learn relies on duck typing (not inheritance), all you need is to create a class and implement three methods: *fit()* (returning self), *transform()*, and *fit_transform()*.

You can get the last one for free by simply adding *TransformerMixin* as a base class. Also, if you add *BaseEstimator* as a base class (and avoid \*args and \**kargs in your constructor) you will get two extra methods (*get_params()* and *set_params()*) that will be useful for automatic hyperparameter tuning. For example, here is a small transformer class that adds the combined attributes we discussed earlier:
```Python
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

In this example the transformer has one hyperparameter, *add_bedrooms_per_room*, set to True by default (it is often helpful to provide sensible defaults). This hyperparameter will allow you to easily find out whether adding this attribute helps the Machine Learning algorithms or not. More generally, you can add a hyperparameter to gate any data preparation step that you are not 100% sure about. The more you automate these data preparation steps, the more combinations you can automatically try out, making it much more likely that you will find a great combination (and saving you a lot of time).
### Feature Scaling
One of the most important transformations you need to apply to your data is *feature scaling*. With few exceptions, Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales. This is the case for the housing data: the total number of rooms ranges from about 6 to 39,320, while the median incomes only range from 0 to 15. Note that scaling the target values is generally not required.

There are two common ways to get all attributes to have the same scale: *min-max scaling* and *standardization*.

Min-max scaling (many people call this *normalization*) is quite simple: values are shifted and rescaled so that they end up ranging from 0 to 1. We do this by subtracting the min value and dividing by the max minus the min. Scikit-Learn provides a transformer called MinMaxScaler for this. It has a feature_range hyperparameter that lets you change the range if you don’t want 0–1 for some reason.

Standardization is quite different: first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the variance so that the resulting distribution has unit variance. Unlike min-max scaling, standardization does not bound values to a specific range, which may be a problem for some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1). However, standardization is much less affected by outliers. For example, suppose a district had a median income equal to 100 (by mistake). Min-max scaling would then crush all the other values from 0–15 down to 0–0.15, whereas standardization would not be much affected. Scikit-Learn provides a transformer called StandardScaler for standardization.
### Transformation pipelines
As you can see, there are many data transformation steps that need to be executed in the right order. Fortunately, Scikit-Learn provides the Pipeline class to help with such sequences of transformations. Here is a small pipeline for the numerical attributes:
```Python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
      ('imputer', Imputer(strategy="median")),
      ('attribs_adder', CombinedAttributesAdder()),
      ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
```
The Pipeline constructor takes a list of name/estimator pairs defining a sequence of steps. All but the last estimator must be transformers (i.e., they must have a *fit_transform()* method). The names can be anything you like.

 When you call the pipeline’s *fit()* method, it calls *fit_transform()* sequentially on all transformers, passing the output of each call as the parameter to the next call, until it reaches the final estimator, for which it just calls the *fit()* method.

The pipeline exposes the same methods as the final estimator. In this example, the last estimator is a *StandardScaler*, which is a transformer, so the pipeline has a trans *form()* method that applies all the transforms to the data in sequence (it also has a *fit_transform* method that we could have used instead of calling *fit()* and then *transform()*).

You now have a pipeline for numerical values, and you also need to apply the *LabelBinarizer* on the categorical values: how can you join these transformations into a single pipeline? Scikit-Learn provides a *FeatureUnion* class for this. You give it a list of transformers (which can be entire transformer pipelines), and when its *transform()* method is called it runs each transformer’s *transform()* method in parallel, waits for their output, and then concatenates them and returns the result (and of course calling its *fit()* method calls all each transformer’s *fit()* method). A full pipeline handling both numerical and categorical attributes may look like this:
```Python
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
      ('selector', DataFrameSelector(num_attribs)),
      ('imputer', SimpleImputer(strategy="median")),
      ('attribs_adder', CombinedAttributesAdder()),
      ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
      ('selector', DataFrameSelector(cat_attribs)),
      ('label_binarizer', MyLabelBinarizer()),
])
full_pipeline = FeatureUnion(transformer_list=[
      ("num_pipeline", num_pipeline),
      ("cat_pipeline", cat_pipeline),
])
```
Due to an error with LabelBinarizer number of arguments we have to create a new class called MyLabelBinarizer like the following:
```Python
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

```
Each subpipeline starts with a selector transformer: it simply transforms the data by selecting the desired attributes (numerical or categorical), dropping the rest, and converting the resulting DataFrame to a NumPy array. There is nothing in Scikit-Learn to handle Pandas DataFrames, so we need to write a simple custom transformer for this task:
```Python
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

```
Now we can run the whole pipeline simply:
```Python
housing_prepared= full_pipeline.fit_transform(housing)
print(housing_prepared)
```
Output:

![Full Pipeline](../img/chp2_full_pipeline.png?raw=true "Full Pipeline")

## Select a model and train it
At last! You framed the problem, you got the data and explored it, you sampled a training set and a test set, and you wrote transformation pipelines to clean up and prepare your data for Machine Learning algorithms automatically. You are now ready to select and train a Machine Learning model.
### Training and Evaluating on the Training Setting
The good news is that thanks to all these previous steps, things are now going to be much simpler than you might think. Let’s first train a Linear Regression model, like we did in the previous chapter:
```Python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```
Done! You now have a working Linear Regression model. Let’s try it out on a few instances from the training set:
```Python
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))
```
Output:

![Predictions](../img/chp2_predictions.PNG?raw=true "Predictions")

It works, although the predictions are not exactly accurate. Let’s measure this regression model’s *RMSE* on the whole training set using Scikit-Learn’s *mean_squared_error* function:
```Python
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
```
Output:

![RMSE](../img/chp2_rmse.png?raw=true "RMSE")

Okay, this is better than nothing but clearly not a great score: most districts’ *median_housing_values* range between $120,000 and $265,000, so a typical prediction error of $68,628 is not very satisfying. **This is an example of a model underfitting the training data**. When this happens it can mean that the features do not provide enough information to make good predictions, or that the model is not powerful enough. As we saw in the previous chapter, the main ways to fix underfitting are to select a more powerful model, to feed the training algorithm with better features, or to reduce the constraints on the model. This model is not regularized, so this rules out the last option. You could try to add more features (e.g., the log of the population), but first let’s try a more complex model to see how it does.

Let’s train a *DecisionTreeRegressor*. This is a powerful model, capable of finding complex nonlinear relationships in the data (Decision Trees are presented in more detail in [Chapter 6]()). The code should look familiar by now:
```Python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
```
Now that the model is trained, let’s evaluate it on the training set:
```Python
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)
```
Output:

![RMSE Tree](../img/chp2_rmse_tree.png?raw=true "RMSE Tree")

Wait, what!? No error at all? Could this model really be absolutely perfect? Of course, it is much more likely that the model has badly overfit the data. How can you be sure? As we saw earlier, you don’t want to touch the test set until you are ready to launch a model you are confident about, so you need to use part of the training set for training, and part for model validation.

### Better Evaluation USing Cross-validation
One way to evaluate the Decision Tree model would be to use the *train_test_split* function to split the training set into a smaller training set and a validation set, then train your models against the smaller training set and evaluate them against the validation set. It’s a bit of work, but nothing too difficult and it would work fairly well.

A great alternative is to use Scikit-Learn’s *cross-validation* feature. The following code performs *K-fold cross-validation*: it randomly splits the training set into 10 distinct subsets called folds, then it trains and evaluates the Decision Tree model 10 times, picking a different fold for evaluation every time and training on the other 9 folds.

The result is an array containing the 10 evaluation scores:
```Python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
```
Let's create a function to show the results:
```Python
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
```
Let's look at the results:
```Python
display_scores(rmse_scores)
```
Output:

![RMSE Tree Score](../img/chp2_rmse_tree_score.png?raw=true "RMSE Tree Score")

Now the Decision Tree doesn’t look as good as it did earlier. In fact, it seems to perform worse than the Linear Regression model! Notice that cross-validation allows you to get not only an estimate of the performance of your model, but also a measure of how precise this estimate is (i.e., its standard deviation). The Decision Tree has a score of approximately 71,200, generally ±3,100. You would not have this information if you just used one validation set. But cross-validation comes at the cost of training the model several times, so it is not always possible.

Let’s compute the same scores for the Linear Regression model just to be sure:
```Python
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)
```
Output:

![RMSE Tree Score](../img/chp2_rmse_lin_score.png?raw=true "RMSE Tree Score")

That’s right: the Decision Tree model is overfitting so badly that it performs worse than the Linear Regression model.

Let’s try one last model now: the *RandomForestRegressor*. As we will see in [Chapter 7](), Random Forests work by training many Decision Trees on random subsets of the features, then averaging out their predictions. Building a model on top of many other models is called *Ensemble Learning*, and it is often a great way to push ML algorithms even further. We will skip most of the code since it is essentially the same as for the other models:
```Python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)
```
Output:

![RMSE Forest](../img/chp2_rmse_forest.png?raw=true "RMSE Forest")

Wow, this is much better: Random Forests look very promising. However, note that the score on the training set is still much lower than on the validation sets, meaning that the model is still overfitting the training set. Possible solutions for overfitting are to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.

However, before you dive much deeper in Random Forests, you should try out many other models from various categories of Machine Learning algorithms (several Support Vector Machines with different kernels, possibly a neural network, etc.), without spending too much time tweaking the hyperparameters. The goal is to shortlist a few (two to five) promising models.
### Save your Data
You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. This will allow you to easily compare scores across model types, and compare the types of errors they make. You can easily save Scikit-Learn models by using Python’s pickle module, or using *sklearn.externals.joblib*, which is more efficient at serializing large NumPy arrays:
```Python
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")
```
## Fine-tune your model
Let’s assume that you now have a shortlist of promising models. You now need to fine-tune them. Let’s look at a few ways you can do that.
### Grid Search
One way to do that would be to fiddle with the hyperparameters manually, until you find a great combination of hyperparameter values. This would be very tedious work, and you may not have time to explore many combinations.

Instead you should get Scikit-Learn’s *GridSearchCV* to search for you. All you need to do is tell it which hyperparameters you want it to experiment with, and what values to try out, and it will evaluate all the possible combinations of hyperparameter values, using cross-validation. For example, the following code searches for the best combination of hyperparameter values for the *RandomForestRegressor*:
```Python
from sklearn.model_selection import GridSearchCV

param_grid = [ {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
```
This *param_grid* tells Scikit-Learn to first evaluate all 3 × 4 = 12 combinations of *n_estimators* and *max_features* hyperparameter values specified in the first dict (don’t worry about what these hyperparameters mean for now; they will be explained in [Chapter 7]()), then try all 2 × 3 = 6 combinations of hyperparameter values in the second dict, but this time with the bootstrap hyperparameter set to False instead of True (which is the default value for this hyperparameter).

All in all, the grid search will explore 12 + 6 = 18 combinations of *RandomForestRegressor* hyperparameter values, and it will train each model five times (since we are using five-fold cross validation). In other words, all in all, there will be 18 × 5 = 90 rounds of training! It may take quite a long time, but when it is done you can get the best combination of parameters like this:
```Python
print(grid_search.best_params_)
```
Output:

![Best Combination](../img/chp2_fine_tune.png?raw=true "Best Combination")

Since 30 is the maximum value of n_estimators that was evaluated,
you should probably evaluate higher values as well, since the
score may continue to improve.

Let's do this and also print the best estimator:
```Python

param_grid = [ {'n_estimators': [3, 10, 60, 100, 200], 'max_features': [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)
```
Output:

![Best Combination 2](../img/chp2_fine_tune2.png?raw=true "Best Combination 2")

Now let's take the *RandomForestRegressor* that it told us to use and print the RMSE to see th differences:
```Python
forest_reg=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features=8, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=200,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)
```
Output:

![Predictions 2](../img/chp2_predictions2.PNG?raw=true "Predictions 2")

## Launch, monitor, and maintain your system
Perfect, you got approval to launch! You need to get your solution ready for production, in particular by plugging the production input data sources into your system and writing tests.

You also need to write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops. This is important to catch not only sudden breakage, but also performance degradation. This is quite common because models tend to “rot” as data evolves over time, unless the models are regularly trained on fresh data.

Evaluating your system’s performance will require sampling the system’s predictions and evaluating them. This will generally require a human analysis. These analysts may be field experts, or workers on a crowdsourcing platform (such as Amazon Mechanical Turk or CrowdFlower). Either way, you need to plug the human evaluation pipeline into your system.

You should also make sure you evaluate the system’s input data quality. Sometimes performance will degrade slightly because of a poor quality signal (e.g., a malfunctioning sensor sending random values, or another team’s output becoming stale), but it may take a while before your system’s performance degrades enough to trigger an alert. If you monitor your system’s inputs, you may catch this earlier. Monitoring the inputs is particularly important for online learning systems.

Finally, you will generally want to train your models on a regular basis using fresh data. You should automate this process as much as possible. If you don’t, you are very likely to refresh your model only every six months (at best), and your system’s performance may fluctuate severely over time. If your system is an online learning system, you should make sure you save snapshots of its state at regular intervals so you can easily roll back to a previously working state.
