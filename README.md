# Kaggle-HumanVRobot

### Final Standing
- Finished 314/985 (top third) in private leaderboard 

### Problem Statement
- The problem was to look at the bidding behavior of a user in online auctions and classify the user as a genuine user or bidding bot. The problem is important because the ability to isolate and suspend bot accounts accurately improves the user experience and may help improve human bidders enrollment and engagement. 
- The problem boils down to a binary classification problem with skewed classes (way more humans than bots) and significant feature engineering prior to model training and prediction

### Files
- eda2.Rmd 
Load in original dataset for competition, exploratory data analysis and feature engineering
- classifier_exp.py -
Load in engineered features dataframe for training and test sets, model training, tuning and prediction and create submission files
- xgboost_clf.Rmd -
XGBoost model for training and prediction in R. Didn't do as well as expected although it was blazing fast.
- ensemble_blending.py -
Create a blended ensemble of best tuned models to improve robustness of final submission and reduce overfitting to training set

### Models & Tuning
- Overall, Random Forests, Extra Trees and Gradient Boosted Trees proved to be the best performing classifiers
- Since ROC AUC was evaluation metric, add scoring = 'roc_auc' to GridSearchCV made all the difference in fine tuning params
- Using cv = 5 or 8 in GridSearchCV instead of default 3 increased the robustness of CV score (less prone to overfitting)
- Using Stratified CV was important because of the skewed positive class to create balanced CV folds

### Feature engineering
- Used a combination of counts, means, stddev, max, median of various features from the bids data for feature engineering, and merchandise
 as a factor in the long form
- Use of dplyr, joins, selects, imputing etc was a great learning experience in this dataset
- As number of features increased, CV ROC_auc trended closer and closer to submission ROC_auc
- Some links state that using 'entropy' as an info gain parameter is more 'brittle' than 'gini' although my entropy models always did slightly better on leaderboard
- Reducing learning rate for GBC to ~0.03-0.05 provided better models than  default of 0.1 (A good value of learning rate maybe more problem specific)
		- However, we have also seen in documentation that smaller learning rates lead to more estimators aka complex model, but more accuracy in general

### Final Model
- Total of 59 features used
- Used tuned Random Forest, Extra Trees and Gradient Boosted Trees blended/stacked together using Logistic Regression
- Limited benefits from CalibrationClassifierCV although using holdout helped me evaluate its effectiveness somewhat

### Did Not Work
- SVM and AdaBoost models did pretty poorly
- Tried to use anomaly detectors to detect humans that may have been mis-classified as robots. This was not a good idea 
- Learned that using PCA or SelectKBest leaves out performance on the table because we are throwing away some variance in the feature set. 
  - For Kaggle like competitions, don't bother with feature selection unless number of features become untenable
- Using class weighting proportional to class ratios for tuning tree-based models proved to be worse than without class weighting which was opposite of what some blogs suggested.
  
###Could have done
- Was unable to use the time column to derive useful features. Based on others sharing models, this was likely the difference between top 10% and 25%
- Could have also used countries as a long form feature similar to merchandise since there were only 200 countries. Failed to do this and might have made a big difference. 

