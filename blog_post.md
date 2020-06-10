# KAGGLE FOOTBALL PREDICTION
### Team members: Mark-Eerik Kodar, Liina Anette Pärtel, Robin Sulg, Karl Riis

# Introduction

The aim of this project is to predict the outcome of a football match.  In short, our goal is to build a preliminary random forest model that can be compared to the final neural network model. 

# Data
The dataset of that the model is trained on can be found here: https://www.kaggle.com/hugomathien/soccer. 
The dataset contains the statistics of over 25 000 matches and 10 000 players from the top divisions of 11 European countries. Each match has list of detailed events from the goal types, possession to fouls and cards. In addition to that, it also provides the betting companies odds for the matches. 


# Data preprocessing
While browsing the dataset it was obvious, that much of the information there was weirdly formatted and some of the data there was really not of use to use. 

Firstly, we disregarded the data that the model should not know beforehand. This meant that in the match information, data such as home team goals, away team goals, shots on target, shots off target, fouls commited, etc. were all removed.

As for the more techincal side of things, we dropped columns that contained null values and joined different tables in the dataset to help us in the later phases of the model development. 

As for the preprocessing steps we have yet to solve, then we noticed that the data is quite imbalanced, as the labels of win, draw, lose are so that there are significantly more win labels and equal amount of draw and lose labels.

<img src='imbalanced_data.png'>

 Also we have thought about normalizing the data, but have yet to decide how to do it. 

# Baseline model

We chose a simple random forest model to act as a baseline. To implement this we chose the sklearn RandomForest model. The initial model without much tuning returned 51.2% accuracy. 

Below is the confusion matrix of the predictions of the model. 

<img src='rf_cf.png'>

# Neural Network

We built our model using Keras with the following architecture:

<img src='keras_layers.png'>

In depth, the first layer uses ReLU as an activation function and uses L2 regularization with the value of 0.0002. The next two layers have the same setup. The fourth layer uses softmax as an activation function and has the same regularization parameter. The optimizer that we use is Adam. The loss function that we use is categorical cross-entropy as this is the de facto standard for multi-class classification problems. 

The current model has an accuracy of 53.2%, but as we can see from the confusion matrix below, then there is a problem with our model, as it does not predict any draws which we'll have to work on later.

<img src='nn_cf.png'>

# Work that still needs to be done:

- Data preprocessing steps mentioned above.
- We have yet to try out and think through the architecture of the neural network. (e.g. the number of layers, dropout, etc. )
- In addition to that we have yet to do hyperparameter optimization.

# Questions that have arisen:
- What causes the NN to "ignore" one class? During several runs of the neural network, we have seen it maybe predicts only a few (up to 5) draws but in most cases it predicts zeros. 
- Would making the labels into one-hot-encoded vectors help?
- What causes the loss value to be exactly the same over several epochs?
