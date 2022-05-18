# supervised-machine-learning-challenge

To view the predictions and results of this challenge, please open the "Credit Risk Evaluator.ipynb" file.

## Background

This assignment looked at an undersampled dataset provided by LendingClub, a peer-to-peer lending services company. The idea is to predict whether a particular loan is high or low risk based off of key features gathered by LendingClub.

## Methods

The key methods used in this assignment are the Logistic Regression and Random Forest Classifiers provided by `scikit-learn` along with the `StandardScaler` from the same library. `Pandas` was used for the one-hot encoding of categorical variables with the `get_dummies()` function. All analysis is my own with no expectation of correct predictions.

**The observations have been recorded below for convenience.**

## Prediction: Logistic Regression vs. Random Forest (Pre-Scaled)

I believe that with a total of 92 features (after encoding categorical data and removing indices), a random forest (of decision trees) will result in a more robust way of classifying the test data. Because logistic regressions take in all features and determines the classifier based on *all* 92 features, it runs the risk of overfitting to the data. In a random forest, because each random combination of features are built out of feature subsets, we can focus only on those features that seem to make the most sense. Without scaling, I anticipate that the random forest will also be better as the huge range in values will potentially bias the outcome of the logistic regression.

## Results: Logistic Regression vs. Random Forest (Pre-Scaled)

Looking at the results of the testing scores, we can see that the random forests did indeed have a better result, but not by much. This is surprising as I felt as though the sheer volume of features that were accounted for in the random forests would influence the outcome of the score more than the logistic regression. On reflection, however, because we only have two results that we are interested in (low-risk vs. high-risk loans), using a logistic regression is not all that inappropriate as we are strictly looking for whether a data point falls on either end of the outcome.

## Prediction: Logistic Regression vs. Random Forest (Scaled)

Based off of the results from earlier, I believe that the logistic regression may edge head of the random forests in terms of score. This is due to the fact that now that the features are being scaled appropriately, we may get a better sense of how the each feature contributes to the overall model. With the logistic regression taking into account all features at once, having the data be scaled eliminates any potential bias due to extreme values. That being said, I do think the logistic regression, while having the "better" score, will still have issues with overfitting due to the sheer volume of features. Not all features are important, and the random forest better accounts for this fact.

## Results: Logistic Regression vs. Random Forest (Scaled)

While I expected the logistic regression score to improve, I did not expect it to improve by so much. I'm also surprised by the fact that the random forest model score barely changed at all. I still believe that this is due to the scaled data being better utilized by the logistic regression, as the random forest already accounted for much of the variability through the ensemble of randomly selected features created in its decision trees.

If I were to pick between the models for further tuning, I would honestly start with the random forest, even though the logistic regression model has a superior score. The reason for this is because I am still uncomfortable with the fact that there could be potential overfitting problems with the logistic regression model. However, it may also be worth trying various hyperparameters in the future in order to get a better understanding of which features are actually important.
