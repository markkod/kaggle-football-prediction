# KAGGLE FOOTBALL PREDICTION
##### Team members: Mark-Eerik Kodar, Liina Anette Pärtel, Robin Sulg, Karl Riis

# Introduction

The aim of this project is to predict the outcome of a football match using a neural network. This neural network model will be compared to a baseline random forest classifier and in the end we will calculate how much money we would end up making or losing if we had bet 1 euro on each game. 

# Data
The dataset of that the model is trained on can be found here: https://www.kaggle.com/hugomathien/soccer. 
The dataset contains the statistics of over 25 000 matches and 10 000 players from the top divisions of 11 European countries. Each match has list of detailed events from the goal types, possession to fouls and cards. In addition to that, it also provides the betting companies' odds for the matches. 


# Data preprocessing
While browsing the dataset it was obvious, that much of the information there was weirdly formatted and some of the data there was really not of use to use. 

Firstly, we disregarded the data that the model should not know beforehand. This meant that in the match information, data such as home team goals, away team goals, shots on target, shots off target, fouls committed, etc. were all removed.

Secondly, we threw away the player data, which was lacking in the sense that there were a lot of null values and also the player data was based on FIFA video game series which may not provide the most accurate estimate of a player's abilities. 

As for the more technical side of things, we dropped columns that contained null values and joined different tables in the dataset to help us in the later phases of the model development. Examples of this include creating features of match win percentage overall, home and away win percentages, win percentage against certain opponent and so on. Also, we noticed that the initial data was quite imbalanced as the labels of home team winning were in the majority as seen below. 

<img src='images/imbalanced.png'>

This problem was solved by oversampling using the <code>imbalanced-learn</code> package. After re-sampling the training dataset we got equal amount of labels for each class.

<img src='images/balanced.png'>

In addition to all that, we needed a way to test our model in a real-life scenario. The way we decided to do that, was to order the dataset by match dates take the last 3 months of matches as our test set on which we can simulate our betting. 

# Neural Network

We started our development of the neural network model by just adding some fully connected layers on top of each other, added L2 regularization, used ReLU activation function and used some fixed learning- and dropout rates. The loss function that we used was categorical cross-entropy as this loss function suits best our multi-class classification task. However, we tried different loss functions as well, e.g. mean square error but stayed with categorical cross-entropy. Categorical cross-entropy loss function also assumes, that the last layer has the same number of hidden nodes that there are the classes and that the last layer uses softmax as an activation function. As for the optimizers we used Stochastic Gradient Descent (SGD) and Adam. SGD performed worse so we stayed with Adam.

This model did not turn out very well even though the validation accuracies were decent (~52%), as this model predicted home wins only. This meant several hours of research of things that could be wrong with our model. This research led us to believe, that our network has maybe too many layers and there might be a problem of ReLU nodes dying. The latter might also be caused too high of a learning rate. Therefore we decided to try whether grid search would help to solve that problem.

Unfortunately, it did not help and after doing some more research we started off from scratch. We decided to now use tanh activation function as with this function there was a smaller probability of neurons dying. We created a model that consisted of just two hidden layers. As seen below, the first layer had 8 units, the tanh activation function and L2 regularization. After that we added dropout and another fully connected layer with softmax activation function.

```python
def create_model(learning_rate=1e-5, dropout_rate=0.1):
    model = Sequential() 
    model.add(Dense(8, input_dim=columns, activation='tanh', kernel_regularizer='l2'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    return model
```

After creating this new model we needed to find the best possible learning rate and dropout rate and the hidden layer size for our model. This was done again using grid search.

```python
learning_rates = [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 1e-6]
dropout_rates = [0.05, 0.1, 0.2]
hidden_sizes = [8, 16, 64, 128]

best_lr = None
best_dr = None
best_model = None
best_hs = None

best_val_acc = 0

for lr in learning_rates:
    for dr in dropout_rates:
        for hs in hidden_sizes:
            model = create_model(lr, dr, hs)
            history = model.fit(
                X_train,
                y_train_categorical,
                epochs=100,
                validation_split=0.1,
                verbose=0)
            val_acc = max(history.history['val_accuracy'])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_dr = dr
                best_hs = hs
                best_model = model
        
```

Initially, we tried to train our model with minibatches of various sizes, but we saw that this produced really poor validation accuracy that was even worse than random in many cases. Therefore modified our grid search so that we trained our model on the whole training dataset, as the initial dataset was not that big and therefore the training times were not that long. During the grid search we saw that the results were significantly better than before. 

Finally we found that the best hyperparameters were as follows: 
- Learning rate: 1e-05
- Dropout rate: 0.1
- Hidden size: 64

We kept the last 3 month match data as the test set. Now using the best model found during grid search we wanted to find out how accurate it would be on the test set. 

The accuracy on the test set was 52.57% and the corresponding confusion matrix is below:

<img src='images/confusion.png'>

# Neural Network vs. Random Forest (baseline)

As it was mentioned above the initial neural network model achieved 52-53% level of accuracy and that was because it predicted everything as home team winning. In addition to the problems in our neural network, part of that was also due to the skewness of the data that we trained the model with. To give the reader a better understanding then almost 50% of the instances in the train set were labelled as “home team won” and the remaining half were distributed as “home team lost” (30%) and “draw” (20%). Because of this skewness, the model was much more inclined to predicting that the home team won which led to good accuracy as whilst splitting the data into the test set it was also more likely that matches with “home team won” ended up in the test set. As it was stated in the data pre-processing chapter we used up-sampling (oversampling) to combat this issue. This way the distribution of the labels in the train dataset was even thus removing the predisposition to predicting “home team won”.

With the skewed data, the accuracies of our classifiers were 52% for the neural network and 54% for the random forest (200 trees in the forest). After sampling the data, the neural network performance dropped but the accuracy of random forest went higher. With tree size of 200 in a forest and tree nodes being expanded until all leaves contain less than 2 samples, test accuracy for the random forest is around 57% and 58% when using 1000 trees. As for the neural network the best accuracy that we saw was ~ 53% but more often it would range from 45 – 50% which is worse than just predicting that home team is going to win.


In conclusion in our comparison Random Forest is superior to our neural network when it comes to predicting football results. This conclusion also appears to be shared more often in the world of Machine Learning. 


# Betting

As we saw above, accuracy is not a good indicator how good a model is. To test the model, we decided to simulate the betting process using the games from last 3 months.

For this, we used an algorithm:
* For every betting agency
    * Start with 0 money
    * For every game in the last 3 months
        * "Bet" (subtract) 1€
        * Check if prediction matches the true label
            * If yes, then multiply the 1€ and the coefficient of the agency and add it to the total

As a baseline, we also tried constant betting
