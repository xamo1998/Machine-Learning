
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
$$
RMSE(X,h)=\sqrt{\frac{1}{m}\sum_{i=1}^m (h(x^{(i)})-y^{(i)})^2}
$$

#### Notations
This equation introduces several very common Machine Learning notations.
- **m** is the number of instances in the dataset you are measuring the RMSE on.
	- *For example*, if you are evaluating the RMSE on a validation set of 2,000 districts,
then m = 2,000.
- **x(i)** is a vector of all the feature values (excluding the label) of the ith instance in
the dataset, and **y(i)** is its label (the desired output value for that instance).
	- *For example*, if the first district in the dataset is located at longitude –118.29°,
latitude 33.91°, and it has 1,416 inhabitants with a median income of $38,372,
and the median house value is $156,400 (ignoring the other features for now),
then:
$$
x^{(1)}=\left(\begin{array}{c}-118.29 \\ 33.91 \\1.416 \\38.372 \end{array}\right)
$$


## Get the data

## Discover and visualize the data to gain insights

## Prepare the data for Machine Learning algorithms

## Select a model and train it

## Fine-tune your model

## Present your solution

## Launch, monitor, and maintain your system
