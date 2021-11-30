# What did I learn?

## Table of contents
+ [Models](#models)
  + [Overfitting and underfitting](#overfitting-and-underfitting-models)
  + [TreeBased](#tree-based)
    + [DecisionTrees](#decisiontreeregressor)
    + [RandomForest](#randomforestregressor)
    + [XGBoost](#xgboost)
  + [Linear Regression](#linear-regression)
  + [Ordinary least square](#linear-regression-ordinary-least-square)
  + [Ridge Regression](#ridge-regression)
  + [Parameter Improvement](#parameter-improvement)
+ [Data cleaning](#data-cleaning)
  + [Missing data](#missing-data)
    + [Impuning data](#impuning-data)
  + [Categorical data](#categorical-data)
    + [Onehot encoding](#onehot-encoding)
    + [Ordinal encoding](#ordinal-encoding)
  + [Scaling and Normalization](#scaling-and-normalization)
    + [Scaling](#scaling)
    + [Normalization](#normalization)
  + [Parsing Data](#parsing-data)
    + [parsing Dates](#parsing-dates)
+ [Feature Engineering](#feature-engineering)
  + [Creating Features](#creating-features)
  + [Clustering Data](#clustering-data)
    + [K-Means](#k-means)

## Models
There are many kind of models in machine learning. This includes Neural networks, Decision trees and much more.

### overfitting and underfitting models.

If a model is trained with not enough data or has not enough leaves (end points of the tree) it is called 'underfitting'.
The model is inaccurate due to a low amount of choices.

On the other hand, if a model has too many leaves or was trained on too much data, it is called 'overfitting'
it results in inaccuracy, because it is fitted too tightly around the train dataset.
It will predict houses from the training set quite accurately, but won't for new ones.

### Tree based
#### DecisionTreeRegressor
For each depth, it splits the path into two, with a boolean condition to determine which path to take. 
This is based on all features given.

Treemodels can be configured with a max-depth, leaf count and more.

#### RandomForestRegressor
The RandomForestRegressor takes multiple DecisionTrees, all with a different amount of leaves, max-depth and other parameters,
and trains all of them. This results in some trees being overfit, and others being underfit and even other being just perfect.
When being asked to predict a value, it will run through all trees and chooses the average outcome of all.

#### XGBoost
XGBoost stands for 'Extreme Gradiant Boost'. It is an ensembling method, where it first trains a model,
then adjusts its parameters to work better and adds it to the ensemble. This process is repeated n times or until the early stop is reached.
Early stopping happens, if enabled and after a certain amount of rounds there was no improvement in the models.
To avoid the model stopping too early, due to a local low, an n-init parameter is set to the minimum amount of round which it has to run at least.

### Linear regression
#### Linear regression (Ordinary least square)
When using linear regression, One generates a slope based on the formular <code>y = &sum;m<sub>i</sub>*x<sub>i</sub> + b</code>
for <code>i</code> being the index of the feature and m being the magnitude of it. and <code>b</code> being the intercept.

While Linear Regression is powerful on small sets with little features, it has problems with overfitting on larger sets of features.

Also, this model has no parameters to tune.

#### Ridge Regression
Ridge Regression is very similar to Linear Regression, but it tries to keep its magnitudes for each feature close to zero.
The 'Force' to keep it close to zero is provided using the <code>alpha</code> parameter. A bigger value for alpha results in <code>m</code> values
closer to zero. A small value will be almost the same as the basic linear regression model.

### Parameter improvement
One way of improving the parameters used in the models, is using Gridsearch.

Gridsearch uses a grid of parameters and tries out each combination of them to find the best one.

It can be used on any Object if it implements a score function.

## Data cleaning
The raw data, one gets as the input is often incomplete or has other flaws, which makes analyzing hard if not impossible.

The following headings explain the issue and possible solutions.

### Missing data
There are two main reasons for data to be missing: 
- There wasn't any data to record or 
- the data was not recorded.

One has to look at the data manually to be able to determine which of the two reasons apply for each column and row.

#### Impuning data
To impune data means to replace missing data with the data, which seems most appropriate.

If there was no data to record, in most cases one replaces it with 0, 'None' or similar. 

If the data exists, but wasn't recorded, there are multiple options for one to replace missing data with:
* the average/mean value of all existing data
* the most frequent data
* the min/max value of existing entries
* Backfilling (taking the next existing value in the list)
* K-nearest-neighbors (The value most frequent in entries which have the same values in the other existing columns)

One should pay attention to the amount of missing data one is replacing. 
At some point, it will fetch bad results, because the mean value or whatever value one chooses is too far off of what the
actual value is. In that case, one should consider dropping the whole column.

### Categorical data
Machine learning algorithms are usually only able to handle numeric values.
Though a lot of data is present in non-numeric/text form. 

#### Onehot encoding
When encoding a column using Onehot, for each unique value in the original column, a new column gets generated,
storing a boolean if the original column contained that value.

#### Ordinal encoding
An ordinal encoder applies a random value to each unique value present in the columns.
Optionally, one can supply the encoder with a list of values present in the columns to rank them upfront.

### Scaling and Normalization
For non-tree-based machine learning algorithms, more extreme values have a greater effect than smaller values.
To avoid that issue, one scales and/or normalizes values

#### Scaling
When one scales data, one differs the range of the values. Most commonly between 0 and 1 or 0 and 100 for positive values
This results in the change of '1' in any numeric value has the same importance.

E.g. When comparing Yen and Dollars. The change of one Yen is less important than one Dollar.
One scales the two to make the change equally important.

**Important:** the distribution will NOT change.

#### Normalization
Normalization is used to put your data into a normal distribution.
Many statistics techniques, especially those with 'Gaussian' in their name assume, that your data is normally distributed.

**Important:** Normalizing data DOES change the distribution.


### Parsing Data
The most common input file format is a CSV (column seperated values) File. This file will only contain Strings.
Even though pandas is able to detect and parse numeric values, other formats won't be automatically.
Therefore, one has to do it oneself.

#### Parsing Dates
One of the most common datatype, which is not parsed automatically is a Date or Datetime.
To parse those, one has to manually tell numpy/pandas which format the date is present and use that to parse all Dates.

**Important:** Take a look at the WHOLE Dataset to check if all entries match the date schema.

## Feature Engineering
Raw data is really nice, but as the name suggests, it is raw. It is somewhat usable, but a lot better if one processes it first.
That is what Feature engineering is. Processing raw data to make it a lot more valuable.

### Creating Features
While Tree based algorithms are great at analyzing single columns of data independent on their scale, they're rather bad at analyzing multiple Columns at once.
E.g. when looking at a recipe, the exact amount of flour doesn't matter as much as the ratio to other ingredients.
But Tree based algorithm is only able to look at the exact amount, if one doesn't provide any more information.
In this case it is helpful to add one or more new features representing the ratios of the ingredients. 

In general, when multiple features are rather about the combination of each other than standing alone by themselves,
it is helpful to add new features containing said relationship.

### Clustering data
Clustering data means grouping data by their proximity when mapping them out by certain features.
Most commonly Latitude and Longitude. But other features in strong correlation work too.

One of the most popular Clustering algorithms is
#### K-Means
K-Means clusters data using the following basic steps:
0. Put K (that's why it is **K**-Means) cluster points randomly on the dataset
1. Assign all Data points to the nearest cluster point.
2. Move the cluster points to minimize the distance to its data points.
3. Go to step 1. unless the cluster points have barely if at all moved in the last n iterations or the maximum amount of iterations has been reached.
4. You're done

The clusters can then be added as a new Feature to the dataset.

## Neural Networks
While Neural Networks are still Machine Learning approaches, they differ a lot from other models, such as Decision Trees.
Instead of making hard, boolean decision (either true or false), they are designed to mimic an organic brain with all its 'fuzziness'.

### Neurons
A Neural Network consists of multiple Neurons. 
Each Neuron is only capable of performing a simple task, such as adding up input values.

#### Activation Functions
Usually, in addition to their core task - adding values - they have a so called 'activation function'.
The activation function helps to moderate the inputs. Here are some examples:
- Stepper: 0 while x is below a threshold 1 if above.

While this seems reasonable, nature rarely has hard edges. Let us look at a more gentle curve:
- Signmoid:  1(1+e<sup>-x</sup>) This produces an output in a range of 0 and 1 (both exclusive) with a gentle curve

But there is also a third commonly used activation function:
- ReLu: max(0, x) This means, the output is 0 unless x > 0. When using this function, you may have to normalize the inputs after each layer.
This is the only one appropriate for Regression tasks, as its output varies has an infinite range [0;infinite[,
while the other functions are restricted to [0;1] and the values 0 and 1.

#### Neuron Layers
As one can see, a single neuron is quite simple. It won't be able to do complex recognition or tasks alone.
That is why one uses multiple Layers of neurons, each containing multiple neurons themselves.
Each Neuron of a Layer is connected to each Neuron of the next Layer. 
Therefore, each Neuron is directly and indirectly dependent on all neurons in previous layers.
The first layer of a neural network is called the 'input layer'. It does not perform any calculations.
The last layer is the 'output layer' as its output is, well, the output.
All layers in between are called 'hidden layers' because they don't directly effect the output nor are they directly effected by the input.
They are hidden from the outside.

#### How does a Neural Network learn?
As mentioned above, the task of a Neuron is to e.g. add up numbers.
Though it doesn't directly do it, but weights the numbers instead.
Each connection between two neurons is added a unique weight value, which the input gets multiplied with.
At first, that weight is randomly chosen, then it gets adjusted according the output error.

##### Errors
The output error is just the difference of the actual output and the desired output, which is given by the training data.
Ok, now we have one error value which we have to distribute over multiple weights to adjust. 
We could split them evenly, but that is prune to falsifying already correct weights. 
Or we could split them unevenly according their fraction of all weights.
That means, if one input has a weight of 1 and another of 2, the second is responsible for 2/3 of the error and the first for 1 third.
This process can be repeated for all output nodes.

##### Backpropagating errors
This concept might work fine for the last layer, where we are certain what our error is, but what about the hidden layers?
As we know the error as well as the weight for each connection, we can combine them in the same way we do for our inputs:
multiply their error by the weight fraction. That way, we get our output error for this node. This process can be repeated for all layers.

### Setting initial weights
In the beginning we said, that the weights are chosen randomly. 
That is only half correct.
They are chosen randomly, but not completely:

Mathematicians have worked out ranges for specific shapes of networks and specific sigmoid functions, but that's, well, quite specific.
A good rule of thumb is to choose the weights normally distributed between <code>-1/&radic; (incoming links)</code> and <code>1/&radic; (incoming links)</code>.
The weights should NEVER be all the same, nor 0!!

Not the same, because the error would be evenly distributed and adjusted, resulting in another evenly distributed set of weights,
catching itself in a loop.

Not zero, because zero cancels the input signal completely, and removes the possibility to adjust the weights.

### Choosing a number of nodes
Each layer has a specific amount of nodes or Neurons. So, how does one choose the correct amount if there even is one

#### Input layer
The number of input nodes for a Neural Network is quite obvious: The number of features you have in your input data.
If you have an image of 28 x 28, you want to have 28*28 = 784 input nodes.

#### Output layer
This depends on the problem you want to solve: 

For a Regression problem, where you are searching for a continues output number, such as a house price,
you want to have just one output node containing said number.

For a Classification problem, you want to have the amount of nodes as the number of Classes you are trying to detect.

#### Hidden layer
This is more complex. The number of nodes is the amount of features you are trying to detect.
The more Neurons per layer you have, the more features are you trying to detect. 
There is indeed no perfect way to choose how many Neurons you want.
Best way is to experiment until you find something which works

