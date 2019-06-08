# Chapter 1: The Machine Learning Landscape

## Types of Machine Learning Systems

There are so many different types of Machine Learning systems:

- [Trained with human supervision](#trained-with-human-supervision-types)

  - [Supervised](#supervised-learning)
  - [Unsupervised](#unsupervised-learning)
  - [Semisupervised](#semisupervised-learning)
  - [Reinforcement Learning](#reinforcement-learning)

- [Can learn incrementally on the fly](#supervised-learning)

  - [Online](#online-learning)
  - [Batch](#batch-learning)

- [How they work](#instance-based-versus-model-based-learning)

  - [Comparing new data points to known data points](#instance-based-learning)
  - [Detect patterns in the training data and build a predictive model](#model-based-learning)

**You can combine them** in any way you like.

### Trained With Human Supervision Types

Here we will see the four types we mentioned before.

#### Supervised Learning

The training data you feed to the algorithm includes the desired solutions, called _labels_.

![Classification](../img/chp1_classification.png?raw=true "Classification")

A typical supervised learning task is **_classification_**. **For example**, in a spam filter you provide many example emails along with their _class_ (spam or ham), and it must learn to classify new emails. Another typical task is to predict a _target_ numeric value, such as the price of a car, given a set of _features_ called _predictors_. This sort of task is called **_regression_**.

![Regression](../img/chp1_regression.png?raw=true "Regression")

#### Unsupervised Learning

The training data is unlabeled. The system tries to learn without a teacher.

![Training set](../img/chp1_unsupervised_training_set.png?raw=true "Training set")

Here are some of the most important unsipervised learning algorithms:

- Clustering

  - k-Means
  - Hierarchical Cluster Analysis (HCA)
  - Expectation Maximization

- Visualization and dimensionality reduction

  - Principal Component Analysis (PCA)
  - Kernel PCA
  - Locally-Linear Embedding (LLE)
  - t-distributed Stochastic Neighbor Embedding (t-SNE)

- Association rule learning

  - Apriori
  - Eclat

#### Semisupervised Learning

Some algorithms can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. This is called _semisupervised learning_. **Example:** Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7\. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just one label per person,4 and it is able to name everyone in every photo, which is useful for searching photos.

#### Reinforcement Learning

_Reinforcement Learning_ is a very different beast. The learning system, called an _agent_ in this context, can observe the environment, select and perform actions, and get _rewards_ in return. It must then learn by itself what is the best strategy, called a _policy_, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

![Reinforcement Learning](../img/chp1_reinforcement_learning.png?raw=true "Reinforcement Learning")

### Can learn incrementally on the fly

Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.

#### Batch learning

In _batch learning_, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called _offline learning_.

#### Online learning

In _online learning_, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called _mini-batches_. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine's main memory (this is called _out-of-core_ learning). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.

![Online Learning](../img/chp1_online_learning.png?raw=true "Online Learning")

### Instance-Based Versus Model-Based Learning

One more way to categorize Machine Learning systems is by how they _generalize_. Most Machine Learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to generalize to examples it has never seen before. Having a good performance measure on the training data is good, but insufficient; the true goal is to perform well on new instances. There are two main approaches to generalization: instance-based learning and model-based learning.

#### Instance-based learning

Following the example of the spam filter, one solutions for the implementation is to flag emails that are identical to known spam emails. This solution is a little restrictive so maybe we could flag emails that are **very similar** to known spam ones. This requires a _measure of similarity_ between two emails. For example, we could count the words they have in common.

#### Model-based learning

Another way to generalize from a set of examples is to build a model of these examples, then use that model to make _predictions_. This is called _model-based learning_.
