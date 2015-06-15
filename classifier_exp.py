"""

Author: Vijay Sathish
Date: 05/10/2015

-- An extension of the working classifier.py to try advanced techniques

"""

# from feature_format import featureFormat, targetFeatureSplit
import csv
import re									# regular expression matching
import math
import numpy as np
import time
import matplotlib.pyplot
from sklearn.feature_selection import SelectKBest as skb
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA as rPca
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer 	# For dealing with sparse features
from sklearn.cross_validation import StratifiedKFold as skf
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier as adab
from sklearn.linear_model import (LogisticRegression, RandomizedLogisticRegression)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
import sklearn.cross_validation as cv

# Outlier detection
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

stabilitySelection = False

outlier_detection = False
manual_handcode = False
select_k = False
k_val = 33  # 37f - 4
header = []

### First define all necessary functions

### Load data from CSV data file
# From python 3.4 onwards, there is no csv.reader.next() and we no longer read in as bytes
def load_data_from_csv (filename, train = True, return_header = False) :
	### Outdated python2.7 code
	# csv_obj = csv.reader(open(filename, 'rb'))			
	# skip the first line as it is the header
	# header = csv_obj.next() 						
	csv_obj = csv.reader(open(filename, 'r'))

	data=[] 																		
	for row in csv_obj: 							
		data.append(row[0:]) 								

	### Convert list to numpy array
	data = np.array(data) 									
	# Extract out the first row as a header
	header = data[0, :]
	print ("header: ", header)
	data = data[1:, :]
	if (train) :
		print ("Train data shape: ", data.shape)
	else :
		print ("Test data shape: ", data.shape)
	if (return_header):
		return data, header
	else:
		return data


### Discard the unwanted columns in the data and populate labels and features
def convert_data_to_label_features (data_frame, train) :
	if (train == True):
		features = data_frame[:, 1:59]
		# features = data_frame[:, 1:22]
		# features = data_frame[:, 1:40]
		print ("Training features shape: ", features.shape)
		print ("Train Features Row[0]: ", features[0, :])
		labels = data_frame[:, 59]
		print ("Training labels shape: ", labels.shape      )
		print ("Train Labels Row[0]: ", labels[0])
		features = features.astype(np.float)
		labels = labels.astype(np.float)
		bidder_ids = data_frame[:, 0]
		return features, labels, bidder_ids
	else:
		features = data_frame[:, 1:]
		# features = data_frame[:, 1:8]
		print ("Testing features shape: ", features.shape)
		print ("Test Features Row[0]: ", features[0, :])
		features = features.astype(np.float)
		return features

### perform standard scaling operation
def perform_standard_scaling (features) :
	scaler = StandardScaler()
	scaler.fit(features)
	print ("Completed Standard Scaler fit!")
	return scaler

### Estimate ROC Area under curve using sklearn for best estimator on training set
def estimate_roc_auc (clf, features, labels) :
	holdout_probs = clf.predict_proba(features)
	roc_auc = roc_auc_score(labels, holdout_probs[:, 1])
	print ("ROC area under curve for holdout set: ", roc_auc)

### Extract features for anomaly detection. Discard category features because those are sparse matrices
def extract_features_for_anomaly_det (data_frame) :
	features = data_frame[:, 1:28]
	bidder_ids = data_frame[:, 0]
	features = features.astype(np.float)
	features = StandardScaler().fit_transform(features)
	print ("(Anomaly Detection) Human features shape: ", features.shape)
	return bidder_ids, features

### Correct training labels outcome based on anomaly detection
def apply_anomaly_correction (features, labels, bidder_ids, anomaly_bidders) :
	bidder_ids = np.array(bidder_ids)
	print (bidder_ids[0])
	print (anomaly_bidders[0])

	filter_features = []
	filter_labels = []

	for i, bidder_id in enumerate(bidder_ids) :
		for j, anomaly_bidder in enumerate(anomaly_bidders) :
			if (anomaly_bidder == '2012c722a90bc47f1abbc8319d0cea51p7l5b'):			# Bidder 231 is definitely NOT a bot
				continue
			if (anomaly_bidder == bidder_id) :
				print ("I got here")
				# labels[i] = 1
				break
		else:
			filter_features.append(features[i, :])
			filter_labels.append(labels[i])
	# count number of bots
	num_bots = 0
	for i, outcome in enumerate(labels):
		if (outcome == 1):
			num_bots +=1 
	print ("Num bots after anomaly correction: ", num_bots)
	filter_features = np.array(filter_features)
	filter_labels = np.array(filter_labels)
	print ("Features shape after anomaly correction: ", filter_features.shape)
	print ("Labels shape after anomaly correction: ", filter_labels.shape)
	
	return filter_features, filter_labels			

### Get feature importances
def get_feature_importances (features, labels) :
	# clf = gbc(random_state = 30, max_depth = 3, n_estimators = 100, min_samples_leaf = 2, min_samples_split = 2, learning_rate = 0.05, subsample = 0.9)
	clf = rfc(random_state = 30, max_depth = 6, n_estimators = 100, min_samples_leaf = 1, min_samples_split = 2, n_jobs = 4, criterion = 'entropy')
	clf.fit(features, labels)
	# print ("Feature Importances: ",  clf.feature_importances_)
	print ("Header", header)
	print ("Feature_Importances: ", sorted(zip(map(lambda x: round(x, 5), clf.feature_importances_), header[1:]), 
             reverse=True))
	return clf

### perform standard scaling operation
def perform_scaling (features, scaling = 'standard') :
	if (scaling == 'standard') :
		print ("Performing standard scaling")
		scaler = StandardScaler()
	else :
		print ("Performing min-max scaling")
		scaler = MinMaxScaler()

	scaler.fit_transform(features)
	print ("Completed %s Scaler fit!" %(scaling))
	return features

### Stability Selection using LogisticRegression
def perform_stability_selection(X_train, y_train, round_id = 0) :
	# Defaults: RandomizedLasso(alpha='aic', scaling=0.5, sample_fraction=0.75, n_resampling=200, n_jobs = 1)
	X_train = perform_scaling (X_train, scaling = 'minmax')
	
	#logistic = LogisticRegression(penalty = 'l2', class_weight = 'auto', max_iter = 1000, random_state = 30)
	#logistic.fit(X_train, y_train)
	print ("Round%d - Stability selection -" %(round_id))
	#print ("Logistic (L1 penalty) Feature_Importances: ", sorted(zip(map(lambda x: round(x, 5), logistic.coef_), header[1:]), 
  #           reverse=True))
	#print ("Logistic Feature_Importances: ", logistic.coef_)

	rlog = RandomizedLogisticRegression(random_state = 30, n_jobs = 3, n_resampling = 400)
	rlog.fit(X_train, y_train)
	print ("Randomized Logistic Feature_Importances: ", rlog.scores_)
	print ("Randomized Logistic Feature_Importances: ", sorted(zip(map(lambda x: round(x, 5), rlog.scores_), header[1:]), 
             reverse=True))

### Experimental function 
# This would be useful if categories was sparse matrix like in the Otto group classification problem
# However, since categories are one-hot instead of sparse, it is of little use to RFC
def fit_transform_cat_with_tfidf (features) :
		cat = features[:, 32:42]
		features = features[:, :32]
		print ("Categories shape: ", cat.shape)
		tfidf = TfidfTransformer(norm = None, smooth_idf = False, sublinear_tf = True)
		features = np.append(features, tfidf.fit_transform(cat).toarray(), axis = 1)
		# train = np.append(train, tfidf.fit_transform(train).toarray(), axis=1)
		print ("Categories transformed shape: ", features.shape)

		# print ("Categories transformed[11]: ", features[11, 32:])
		return features, tfidf


def transform_cat_with_tfidf (features, tfidf) :
		cat = features[:, 32:42]
		features = features[:, :32]
		print ("Categories shape: ", cat.shape)
		features = np.append(features, tfidf.transform(cat).toarray(), axis = 1)
		# train = np.append(train, tfidf.fit_transform(train).toarray(), axis=1)
		print ("Categories transformed shape: ", features.shape)

		return features


### Gradient Boosting
def train_model_gbc (features, labels) :
	# Start with reduced param space
	#params_dict = {'n_estimators':[ 50, 60, 70, 80, 90], 'max_depth':[3], 'min_samples_leaf': [1, 2], 'learning_rate': [0.04, 0.05, 0.06], 'min_samples_split': [2, 5, 10], 'subsample': [0.8, 0.9, 1]}
	# params_dict = {'n_estimators':[70, 80, 90], 'max_depth':[3, 4], 'learning_rate': [0.03, 0.04, 0.05], 'subsample': [0.7, 0.8, 0.9], 'max_features': ['sqrt'], 'min_samples_leaf': [1, 2], 'min_samples_split': [2, 5, 10]}
	params_dict = {'n_estimators':[60, 80, 100], 'max_depth':[4, 5, 6], 'learning_rate': [0.03, 0.05, 0.07], 'subsample': [0.5, 0.7, 0.9]}
	
	### Train estimator (initially only on final count
	clf = GridSearchCV(gbc(random_state = 30), params_dict, n_jobs = 4, scoring = 'roc_auc', cv = 5)
	clf.fit(features, labels)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best grid scores: %.4f" %(clf.best_score_))
	return clf

### Gradient Boosted Classifier (Calibrated model with cross validation)
# In this mode, we pre-select features via GridSearchCV, then run calibration on the entire training set since we have a limited dataset
def train_model_gbc_calibrated_cv (features, labels, hold_out = False, train_sz = 0.9) :
	features_train, features_test = [], []
	labels_train, labels_test = [], []
	if (hold_out == True) :
		# First, set aside a some of the training set for calibration
		# Use stratified shuffle split so that class ratios are maintained after the split
		splitter = StratifiedShuffleSplit(labels, n_iter = 1, train_size = train_sz, random_state = 30)

		# Length is 1 in this case since we have a single fold for splitting
		print (len(splitter))

		for train_idx, test_idx in splitter:
			features_train, features_test = features[train_idx], features[test_idx]
			labels_train, labels_test = labels[train_idx], labels[test_idx]
	else :
		features_train = features
		labels_train = labels

	print ("features_train shape: ", features_train.shape)
	print ("labels_train shape: ", labels_train.shape)
	if (hold_out == True) :
		print ("features_test shape: ", features_test.shape)
		print ("labels_test shape: ", labels_test.shape)
		
	print ("Parameters selected based on prior grid Search ...")
	# clf = gbc(random_state = 30, max_depth = 4, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 80, learning_rate = 0.03, subsample = 0.8, max_features = 'sqrt')
	# clf = gbc(random_state = 30, max_depth = 3, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 120, learning_rate = 0.03, subsample = 0.8)
	clf = gbc(random_state = 30, max_depth = 4, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 80, learning_rate = 0.03, subsample = 0.5)

	# Perform calibration 
	# Use 'sigmoid' because sklearn cautions against using 'isotonic' for lesser than 1000 calibration samples as it can result in overfitting
	# 05/22 - Looks like isotonic does better than sigmoid for both Brier score and roc_auc_score.
	# Using 30-40% holdout actually improves ROC AUC for holdout score from 0.88 to 0.925 with CV=5
	print ("Performing Calibration now ...")
	# sigmoid = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
	sigmoid = CalibratedClassifierCV(clf, cv=5, method='isotonic')
	sigmoid.fit(features_train, labels_train)

	if (hold_out == True) :
		# Calculate Brier score loss
		y_probs = sigmoid.predict_proba(features_test)[:, 1]
		clf_score = brier_score_loss(labels_test, y_probs)
		print ("Brier score: ", clf_score)
		auc_score = estimate_roc_auc (sigmoid, features_test, labels_test)

	return sigmoid


### Random Forests Classifier (Calibrated model with cross validation)
# In this mode, we pre-select features via GridSearchCV, then run calibration on the entire training set since we have a limited dataset
def train_model_rfc_calibrated_cv (features, labels, hold_out = False, train_sz = 0.9) :
	features_train, features_test = [], []
	labels_train, labels_test = [], []
	if (hold_out == True) :
		# First, set aside a some of the training set for calibration
		# Use stratified shuffle split so that class ratios are maintained after the split
		splitter = StratifiedShuffleSplit(labels, n_iter = 1, train_size = train_sz, random_state = 30)

		# Length is 1 in this case since we have a single fold for splitting
		print (len(splitter))

		for train_idx, test_idx in splitter:
			features_train, features_test = features[train_idx], features[test_idx]
			labels_train, labels_test = labels[train_idx], labels[test_idx]
	else :
		features_train = features
		labels_train = labels

	print ("features_train shape: ", features_train.shape)
	print ("labels_train shape: ", labels_train.shape)
	if (hold_out == True) :
		print ("features_test shape: ", features_test.shape)
		print ("labels_test shape: ", labels_test.shape)
		
	print ("Parameters selected based on prior grid Search ...")
	#clf = rfc(random_state = 30, n_jobs = 4, criterion = 'entropy', max_depth = 7, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 50)
	#clf = rfc(random_state = 30, n_jobs = 4, criterion = 'gini', max_depth = 8, min_samples_leaf = 5, min_samples_split = 2, n_estimators = 120)
	# clf = rfc(random_state = 30, n_jobs = 4, criterion = 'gini', class_weight = 'auto', max_depth = 5, min_samples_leaf = 5, min_samples_split = 2, n_estimators = 100)
	clf = rfc(random_state = 30, n_jobs = 4, criterion = 'entropy', class_weight = 'auto', max_depth = 5, min_samples_leaf = 5, min_samples_split = 2, n_estimators = 60)

	# Perform calibration 
	# Use 'sigmoid' because sklearn cautions against using 'isotonic' for lesser than 1000 calibration samples as it can result in overfitting
	# 05/22 - Looks like isotonic does better than sigmoid for both Brier score and roc_auc_score.
	# Using 30-40% holdout actually improves ROC AUC for holdout score from 0.88 to 0.925 with CV=5
	print ("Performing Calibration now ...")
	# sigmoid = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
	sigmoid = CalibratedClassifierCV(clf, cv=5, method='isotonic')
	sigmoid.fit(features_train, labels_train)

	if (hold_out == True) :
		# Calculate Brier score loss
		y_probs = sigmoid.predict_proba(features_test)[:, 1]
		clf_score = brier_score_loss(labels_test, y_probs)
		print ("Brier score: ", clf_score)
		auc_score = estimate_roc_auc (sigmoid, features_test, labels_test)

	return sigmoid


### Random Forests Classifier (Calibrated model)
def train_model_rfc_calibrated (features, labels) :
	# First, set aside a some of the training set for calibration
	# Use stratified shuffle split so that class ratios are maintained after the split
	splitter = StratifiedShuffleSplit(labels, n_iter = 1, train_size = 0.7, random_state = 30)

	# Length is 1 in this case since we have a single fold for splitting
	print (len(splitter))

	for train_idx, calib_idx in splitter:
		features_train, features_calib = features[train_idx], features[calib_idx]
		labels_train, labels_calib = labels[train_idx], labels[calib_idx]

	print ("features_train shape: ", features_train.shape)
	print ("features_calib shape: ", features_calib.shape)
	print ("labels_train shape: ", labels_train.shape)
	print ("labels_calib shape: ", labels_calib.shape)
		
	print ("Performing Grid Search ...")
	# params_dict = {'criterion': ['entropy'], 'n_estimators':[30, 35, 40, 45], 'max_depth':[5, 6], 'min_samples_leaf': [1, 2, 5], 'min_samples_split': [2, 5, 10]}
	params_dict = {'criterion': ['entropy'], 'n_estimators':[60, 70, 80, 90], 'max_depth':[5, 6], 'min_samples_leaf': [1, 2, 5], 'min_samples_split': [2, 5, 10], 'max_features' : [6, 7, 8]}
	clf = GridSearchCV(rfc(random_state = 30, n_jobs = 4), params_dict, scoring = 'roc_auc', cv = 5)
	clf.fit(features_train, labels_train)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best best scores: %.4f" %(clf.best_score_))
	# print ("Best grid scores: ", clf.grid_scores_)

	# Perform calibration 
	# Use 'sigmoid' because sklearn cautions against using 'isotonic' for lesser than 1000 calibration samples as it can result in overfitting
	print ("Performing Calibration now ...")
	sigmoid = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
	sigmoid.fit(features_calib, labels_calib)
	return sigmoid

### Extra Trees Classifier (Calibrated model with cross validation)
# In this mode, we pre-select features via GridSearchCV, then run calibration on the entire training set since we have a limited dataset
def train_model_etc_calibrated_cv (features, labels, hold_out = False, train_sz = 0.9) :
	features_train, features_test = [], []
	labels_train, labels_test = [], []
	if (hold_out == True) :
		# First, set aside a some of the training set for calibration
		# Use stratified shuffle split so that class ratios are maintained after the split
		splitter = StratifiedShuffleSplit(labels, n_iter = 1, train_size = train_sz, random_state = 30)

		# Length is 1 in this case since we have a single fold for splitting
		print (len(splitter))

		for train_idx, test_idx in splitter:
			features_train, features_test = features[train_idx], features[test_idx]
			labels_train, labels_test = labels[train_idx], labels[test_idx]
	else :
		features_train = features
		labels_train = labels

	print ("features_train shape: ", features_train.shape)
	print ("labels_train shape: ", labels_train.shape)
	if (hold_out == True) :
		print ("features_test shape: ", features_test.shape)
		print ("labels_test shape: ", labels_test.shape)
		
	print ("Parameters selected based on prior grid Search ...")
	clf = etc(random_state = 30, n_jobs = 4, criterion = 'entropy', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 375)
	# clf = etc(random_state = 30, n_jobs = 4, criterion = 'entropy', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 350)

	# Perform calibration 
	# Use 'sigmoid' because sklearn cautions against using 'isotonic' for lesser than 1000 calibration samples as it can result in overfitting
	# 05/22 - Looks like isotonic does better than sigmoid for both Brier score and roc_auc_score.
	# Using 30-40% holdout actually improves ROC AUC for holdout score from 0.88 to 0.925 with CV=5
	print ("Performing Calibration now ...")
	# sigmoid = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
	sigmoid = CalibratedClassifierCV(clf, cv=5, method='isotonic')
	sigmoid.fit(features_train, labels_train)

	if (hold_out == True) :
		# Calculate Brier score loss
		y_probs = sigmoid.predict_proba(features_test)[:, 1]
		clf_score = brier_score_loss(labels_test, y_probs)
		print ("Brier score: ", clf_score)
		auc_score = estimate_roc_auc (sigmoid, features_test, labels_test)

	return sigmoid

### Extra Trees Classifier
def train_model_etc (features, labels) :
	# Start with reduced param space
	params_dict = {'criterion': ['entropy'], 'n_estimators':[375, 400, 425], 'min_samples_leaf':[1, 2, 5, 10], 'min_samples_split': [2, 5, 10, 20]}
	# params_dict = {'criterion': ['entropy', 'gini'], 'n_estimators':[200, 250, 300, 350]}
	
	### Train estimator (initially only on final count)
	clf = GridSearchCV(etc(random_state = 30, n_jobs = 4, verbose = 0), params_dict, scoring = 'roc_auc', cv = 5)
	clf.fit(features, labels)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best best scores: %.4f" %(clf.best_score_))
	#print ("Best grid scores: ", clf.grid_scores_)
	return clf

### K Nearest Neighbors Classifier
def train_model_knc (features, labels) :
	# Scaling is very important for distance based classifiers
	scaler = StandardScaler()
	clf_knc = knc()

	# Transforms are applied exactly in the order specified
	estimators = [('sscaler', scaler), ('knc', clf_knc)]
	# p = 2 corresponds to Euclidean distance, p = 1 corresponds to Manhattan distance
	params_dict = {'knc__n_neighbors': [5, 8, 10, 15, 20, 25, 30], 'knc__weights':['uniform', 'distance'], 'knc__p': [1, 2]}
	
	clf = GridSearchCV(Pipeline(estimators), params_dict, scoring = 'roc_auc', cv = 5)
	clf.fit(features, labels)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best best scores: %.4f" %(clf.best_score_))
	#print ("Best grid scores: ", clf.grid_scores_)
	return clf

### Random Forests Classifier
def train_model_rfc (features, labels) :
	# Start with reduced param space
	# Best came in at the higher end of 1000, 6, so increase
	# params_dict = {'criterion': ['entropy'], 'n_estimators':[40, 60, 80, 100], 'max_depth':[5, 6, 7], 'min_samples_leaf': [1, 2, 5], 'min_samples_split': [2, 5, 10], 'max_features' : [6, 7]}
	params_dict = {'class_weight' : ['auto'],  'criterion': ['entropy'], 'n_estimators':[50, 60, 70], 'max_depth':[4, 5, 6], 'min_samples_leaf': [1, 2, 5], 'min_samples_split': [2, 5, 10]}

	# params_dict = {'criterion': ['entropy'], 'n_estimators':[100, 150, 200, 250, 300], 'max_depth':[None], 'min_samples_split': [1, 2, 5], 'max_features': [6, 7, 8, 9]}
	
	### Train estimator (initially only on final count
	# skf = StratifiedKFold
	clf = GridSearchCV(rfc(random_state = 30, n_jobs = 4), params_dict, scoring = 'roc_auc', cv = 5)
	clf.fit(features, labels)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best best scores: %.4f" %(clf.best_score_))
	#print ("Best grid scores: ", clf.grid_scores_)
	return clf

### Bagging classifier using Random Forests as the base classifier
def train_model_bagging (features, labels) :
	base_model = rfc(n_estimators = 80, max_features = 20,
                      max_depth=6, random_state = 30,
                      criterion = 'entropy')
	# model = BaggingClassifier(base_estimator = base_model)
	params_dict = {'max_features': [0.5, 0.8], 'max_samples': [0.5, 0.8, 1], 'n_estimators':[25, 50, 75]}
	
	clf = GridSearchCV(BaggingClassifier(random_state = 30, n_jobs = -1, base_estimator = base_model), params_dict, scoring = 'roc_auc', cv = skf(labels, n_folds = 5, random_state = 30))
	clf.fit(features, labels)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best best scores: %.4f" %(clf.best_score_))
	return clf

### ADABoost stacking classifier using Random Forests as the base classifier
def train_model_adab_stacked_rfc (features, labels) :
	base_model = rfc(n_estimators = 80, max_features = 7,
                      max_depth=6, random_state = 30,
                      criterion = 'entropy')
	# model = BaggingClassifier(base_estimator = base_model)
	params_dict = {'learning_rate' : [0.03, 0.05, 0.1], 'n_estimators':[20, 50, 100]}
	
	clf = GridSearchCV(adab(random_state = 30, base_estimator = base_model), params_dict, n_jobs = -1, scoring = 'roc_auc', cv = 5)
	clf.fit(features, labels)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best best scores: %.4f" %(clf.best_score_))
	return clf


### SVC with feature scaler pipeline
def train_model_rfc_pipeline (features, labels) :
	scaler = StandardScaler()
	clf_rfc = rfc(random_state = 30, n_jobs = 4, criterion = 'entropy')

	# Transforms are applied exactly in the order specified
	estimators = [('sscaler', scaler), ('rfc', clf_rfc)]

	t0 = time.clock()
	
	# Use pipeline directly in GridSearchCV
	params_dict = {'rfc__n_estimators': [100, 300, 500, 700], 'rfc__max_depth': [1, 2, 3], 'rfc__min_samples_split':[10, 20, 50], 'rfc__min_samples_leaf':[1, 2, 5]}
	clf = GridSearchCV(Pipeline(estimators), params_dict, cv = 5, scoring = 'roc_auc')
	clf.fit(features, labels)

	print ("Grid Search CV time: ", time.clock() - t0 )
	print ("Best estimator: ", clf.best_estimator_)
	print ("Best grid scores: %.4f" %(clf.best_score_))
	return clf


### Random Forests Classifier
def train_model_adab (features, labels) :
	# Start with reduced param space
	# 140 for learning rate of 0.02
	# 200 for learning rate of 0.015
	params_dict = {'n_estimators':[100, 150, 200, 300, 350], 'learning_rate':[0.03, 0.05, 0.07]}
	
	### Train estimator (initially only on final count
	clf = GridSearchCV(adab(random_state = 30), params_dict, n_jobs = 4, cv = 5, scoring = 'roc_auc')
	clf.fit(features, labels)

	print ("Best estimator: ", clf.best_estimator_)
	print ("Best est score: %.4f" %(clf.best_score_))
	#print ("Best grid scores: ", clf.grid_scores_)
	return clf


### SVC with feature scaler pipeline
def train_model_svc_pipeline (features, labels) :
	scaler = StandardScaler()
	# Setting probability = True is important since we are calculating ROC AUC, but this is also known to slow down the kernel
	clf_svc = SVC(random_state = 30, probability = True, cache_size = 1024)

	# Transforms are applied exactly in the order specified
	estimators = [('sscaler', scaler), ('svc', clf_svc)]
	# estimators = [('mmscaler', MinMaxScaler()), ('svc', clf_svc)]					# Its perf. is lower than StandardScaler as I expected

	# scaler_svc = Pipeline(estimators)
	# scaler_svc.set_params(svc__C = 100, svc__gamma = 0.1, svc__kernel = 'rbf')
	# print ("scaler_svc.fit() ...")
	# scaler_svc.fit(features, labels)

	t0 = time.clock()
	
	# Use pipeline directly in GridSearchCV
	# Setting class_weight = 'auto' is very important for skewed classes
	# params_dict = {'svc__C': [1, 3, 10, 100, 1000], 'svc__gamma':[0.001, 0.01, 0.1, 1], 'svc__kernel':['linear', 'rbf']}
	# params_dict = {'svc__class_weight': ['auto'], 'svc__C': [0.9, 1, 10], 'svc__gamma':[2, 2.3, 2.6, 3], 'svc__kernel':['rbf'], 'svc__tol': [0.0001]}
	params_dict = {'svc__class_weight': ['auto'], 'svc__C': [0.9, 1, 10], 'svc__gamma':[0.1, 0.33, 0.67, 1], 'svc__kernel':['rbf'], 'svc__tol': [0.0001]}
	# params_dict = {'svc__class_weight': ['auto', {0: 1, 1: 20}, {0: 1, 1: 50}], 'svc__C': [1, 10, 30], 'svc__gamma':[0.1, 0.33, 0.67, 1], 'svc__kernel':['rbf']}
	clf = GridSearchCV(Pipeline(estimators), params_dict, n_jobs = 3, cv = 5, scoring = 'roc_auc')
	clf.fit(features, labels)

	# return scaler_svc
	print ("Grid Search CV time: ", time.clock() - t0 )
	# print ("Num support vectors: ", clf_svc.n_support_)
	print ("Best estimator: ", clf.best_estimator_)
	print ("Best grid scores: %.4f" %(clf.best_score_))
	return clf



if __name__ == '__main__':

	############### OUTLIER DETECTION ###############
	if (outlier_detection) :
		# humans_data = load_data_from_csv("D:/Kaggle/HumanVRobot/train_humans_ef_38f.csv", train = True)
		humans_data = load_data_from_csv("D:/Kaggle/HumanVRobot/train_humans_ef_21f_selrlr.csv", train = True)
	
		# Discard category information because it is a sparse matrix and only consider top 28 features
		bidder_ids, features = extract_features_for_anomaly_det (humans_data)

		# clf = OneClassSVM(nu = 0.0025, gamma = 0.0001)
		clf = OneClassSVM(nu = 0.0005, gamma = 0.0033)
		clf.fit(features)
		# clf.decision_function(features)
		pred = np.array(clf.predict(features))
		num_outliers = 0
		outlier_idx = []
		anomaly_bidders = []
		if (manual_handcode == False):
			for i, p in enumerate(pred) :
				if (p == -1):
					num_outliers += 1
					outlier_idx.append([i])
					anomaly_bidders.append(bidder_ids[i])
					# print (" i = ", i, features[i, :])
		else: 	
			print ("WARNING: Handcoding anomaly indices!")
			outlier_idx = [1079, 1807, 184, 564, 1228, 1497]								# These look bot-ish by manual inspeection
			for idx in outlier_idx: 
				anomaly_bidders.append(bidder_ids[idx])
		print ("Num outliers: ", num_outliers)
		print ("Outlier indices: ", outlier_idx)
		print ("Outlier bidder_ids: ", anomaly_bidders)

	############### TRAINING DATA ####################
	### Load data from CSV data file
	# Add feature nlastbids1 (nlastbids was faulty)
	# train_data = load_data_from_csv("D:/Kaggle/HumanVRobot/train_ef_28f_3.csv", train = True)
	# Add additional feature avg_bid_freq
	# train_data, header = load_data_from_csv("D:/Kaggle/HumanVRobot/train_ef_21f_selrlr.csv", train = True, return_header = True)
	train_data, header = load_data_from_csv("D:/Kaggle/HumanVRobot/train_ef_59f.csv", train = True, return_header = True)

	features, labels, bidder_ids = convert_data_to_label_features (train_data, True)
	print ("Post float conversion: ", features[5])

	if outlier_detection:
		features, labels = apply_anomaly_correction (features, labels, bidder_ids, anomaly_bidders)

	### SelectKBest
	if select_k == True:
		print ("***WARNING***: USING SELECT K BEST with k = %d!" %(k_val));
		selector = skb(k = k_val)
		features = selector.fit_transform(features, labels)

	### Stability Selection
	if stabilitySelection:
		perform_stability_selection(features, labels)


	### Fit - Transform sparse matrix
	# features, tfidf = fit_transform_cat_with_tfidf (features)
	
	### Fit Model
	# clf = get_feature_importances (features, labels)
	# clf = train_model_gbc (features, labels)
	# clf = train_model_etc (features, labels)
	# clf = train_model_knc (features, labels)
	# clf = train_model_rfc (features, labels)
	clf = train_model_bagging (features, labels)
	# clf = train_model_adab_stacked_rfc (features, labels)
	# clf = train_model_rfc_calibrated_cv (features, labels)
	# clf = train_model_rfc_calibrated_cv (features, labels, hold_out = True, train_sz = 0.6)    # In case we want to check Brier score, hold out some portion of the training set
	# clf = train_model_gbc_calibrated_cv (features, labels)
	# clf = train_model_gbc_calibrated_cv (features, labels, hold_out = True, train_sz = 0.6)    # In case we want to check Brier score, hold out some portion of the training set
	# clf = train_model_etc_calibrated_cv (features, labels)
	# clf = train_model_etc_calibrated_cv (features, labels, hold_out = True, train_sz = 0.6)    # In case we want to check Brier score, hold out some portion of the training set
	# clf = train_model_rfc_pipeline (features, labels)
	# clf = train_model_adab (features, labels)
	# clf = train_model_svc (features, labels)
	# clf = train_model_svc_pipeline (features, labels)

	############### TESTING DATA ####################
	### Load data from CSV data file
	# Add additional feature avg_bid_freq
	# Add additional feature avg_bid_freq
	# 59 features does not include avg_bid_frac or nlastbids or last_bid_freq; All min features also removed
	# test_data = load_data_from_csv("D:/Kaggle/HumanVRobot/test_ef_21f_selrlr.csv", train = False)
	test_data = load_data_from_csv("D:/Kaggle/HumanVRobot/test_ef_59f.csv", train = False)

	features = convert_data_to_label_features (test_data, train = False)

	### Record bidder_ids for output submission
	bidder_ids = test_data[:, 0]
     
	### Convert features to float
	# features = features.astype(np.float)
	
	### Transform sparse matrix
	# features = transform_cat_with_tfidf (features, tfidf)
	
	### SelectKBest
	if select_k == True:
		features = selector.transform(features)


	### Predict output
	predict_prob = clf.predict_proba(features)

	# open in byte mode for older Python2.7
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/temp.csv', 'wb'))

	### Dump to csv
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/gbc_tune_59_to_40feat_ccv5.csv', 'w', newline = ''))
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/rfc_tune_59_to_40feat_ccv5.csv', 'w', newline = ''))
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/etc_gini_tune_59feat_ccv5.csv', 'w', newline = ''))
	csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/temp.csv', 'w', newline = ''))
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/bagging_tune_59feat_cv5.csv', 'w', newline = ''))

	data = []
	data.append(['bidder_id', 'prediction'])
	for idx in range(len(predict_prob)):
		data.append([bidder_ids[idx], predict_prob[idx, 1]]) 					# We want to only dump class probabilities for bot

	### Need to add in bidder_ids with no bids and predict outcome 0
	test_data_orig = load_data_from_csv("D:/Kaggle/HumanVRobot/test.csv", train = False)
	bidder_ids_orig = test_data_orig[:, 0]
	no_bids = []
	for bid_id1 in bidder_ids_orig:
		for bid_id2 in bidder_ids:
			# print (bid_id1, bid_id2)
			if (bid_id1 == bid_id2):
				break
		else:
			no_bids.append(bid_id1)

	no_bids_np = np.array(no_bids)
	print ("Shape of no_bids: ", no_bids_np.shape)

	for idx in range(len(no_bids)):
		data.append([no_bids[idx], 0.0]) 					# We want to only dump class probabilities for bot

	csv_out.writerows(data)


