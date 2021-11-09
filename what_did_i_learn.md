# What did I learn?

## Table of contents
+ [Models](#models)
  + [Overfitting and underfitting](#overfitting-and-underfitting-models)
  + [TreeBased](#tree-based)
    + [DecisionTrees](#decisiontreeregressor)
    + [RandomForest](#randomforestregressor)
    + [XGBoost](#xgboost)
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
