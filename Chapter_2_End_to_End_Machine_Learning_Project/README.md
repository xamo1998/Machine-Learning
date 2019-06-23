
# Chapter 2: End to End Machine Learning Project
In this chapter we will create a full project, first we will see the main steps we will be doing:
1. [Look at the big picture](#look-at-the-big-picture)
2. [Get the data](#get-the-data)
3. [Discover and visualize the data to gain insights](#discover-and-visualize-the-data-to-gain-insights)
4. [Prepare the data for Machine Learning algorithms](#prepare-the-data-for-machine-learning-algorithms)
5. [Select a model and train it](#select-a-model-and-train-it)
6. [Fine-tune your model](#fine-tune-your-model)
7. [Present your solution](#present-your-solution)
8. [Launch, monitor, and maintain your system](#launch-monitor-and-maintain-your-system)

## Data we will be using
the California Housing Prices dataset from the StatLib repository. This dataset was based on data from the 1990 California census. It is not exactly recent, but it has many qualities for learning, so we will pretend it is recent data. We
also added a categorical attribute and removed a few features for teaching purposes.
## Look at the big picture
The first task you are asked to perform is to build a model of housing prices in California using the California census
data.
Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.
### Frame the Problem
The first question to ask is what exactly is the business objective; building a model is probably not the end goal. This is important because it will determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it.
Our model’s output (a prediction of a district’s median housing price) will be fed to another Machine Learning system, along with many other signals. This downstream system will determine whether it is worth investing in a given area or not. Getting this right is critical, as it directly affects revenue.

![Pipeline](../img/chp2_pipeline.png?raw=true "Pipeline")

#### Pipelines
A sequence of data processing components is called a data pipeline. Pipelines are very common in Machine Learning systems, since there is a lot of data to manipulate and many data transformations to apply.

First, you need to frame the problem: is it *supervised*, *unsupervised*, or *Reinforcement Learning*? Is it a *classification* task, a *regression* task, or something else? Should you use *batch learning* or *online learning* techniques?
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


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;And:
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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ℓ<sub>0</sub> just gives the cardinality of the vector (i.e., the number of elements), and ℓ<sub>∞</sub> gives the maximum absolute value in the vector.
 - The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.


## Get the data
The code will be using can be found in [ttps://github.com/ageron/handson-ml](ttps://github.com/ageron/handson-ml).
### Create the Workspace
Make sure you have everything installed as we saw in the [Installation](https://github.com/xamo1998/Machine-Learning#installation) guide.
### Downloading the Data
## Discover and visualize the data to gain insights

## Prepare the data for Machine Learning algorithms

## Select a model and train it

## Fine-tune your model

## Present your solution

## Launch, monitor, and maintain your system
