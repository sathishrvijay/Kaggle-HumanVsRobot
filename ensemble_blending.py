"""
"You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)"
The predictions are saved in test.csv. 

Original codebase from:
https://github.com/emanuele/kaggle_pbr/blob/master/blend.py

"""

from __future__ import division
import numpy as np
import csv
import re									# regular expression matching
import math
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import AdaBoostClassifier as adab

### Load data from CSV data file
# From python 3.4 onwards, there is no csv.reader.next() and we no longer read in as bytes
def load_data_from_csv (filename, train = True) :
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
	return data

### Discard the unwanted columns in the data and populate labels and features
def convert_data_to_label_features (data_frame, train) :
	if (train == True):
		features = data_frame[:, 1:22]
		# features = data_frame[:, 1:43]
		# features = data_frame[:, 1:38]
		print ("Training features shape: ", features.shape)
		print ("Train Features Row[0]: ", features[0, :])
		labels = data_frame[:, 22]
		print ("Training labels shape: ", labels.shape      )
		print ("Train Labels Row[0]: ", labels[0])
		features = features.astype(np.float)
		labels = labels.astype(np.float)
		return features, labels
	else:
		features = data_frame[:, 1:]
		# features = data_frame[:, 1:8]
		print ("Testing features shape: ", features.shape)
		print ("Test Features Row[0]: ", features[0, :])
		features = features.astype(np.float)
		return features



if __name__ == '__main__':

	np.random.seed(0) # seed to shuffle the train set

	n_folds = 5
	calib_folds = 3
	verbose = True
	shuffle = False

	# Returns Training features, training labels and testing features in that order
	# X, y, X_submission = load_data.load()
	# train_data = load_data_from_csv("D:/Kaggle/HumanVRobot/train_ef_38f.csv", train = True)
	train_data = load_data_from_csv("D:/Kaggle/HumanVRobot/train_ef_21f_selrlr.csv", train = True)
	X, y = convert_data_to_label_features (train_data, True)

	# test_data = load_data_from_csv("D:/Kaggle/HumanVRobot/test_ef_38f.csv", train = False)
	test_data = load_data_from_csv("D:/Kaggle/HumanVRobot/test_ef_21f_selrlr.csv", train = False)
	X_submission = convert_data_to_label_features (test_data, False)

	### Record bidder_ids for output submission
	bidder_ids = test_data[:, 0]
     
	if shuffle:
		idx = np.random.permutation(y.size)
		X = X[idx]
		y = y[idx]

	skf = list(StratifiedKFold(y, n_folds))

	### Setup the SVC pipeline
	scaler = StandardScaler()
	mmScaler = MinMaxScaler()
	clf_svc = SVC(random_state = 30, probability = True, cache_size = 1024, kernel = 'rbf', gamma = 2.60, C = 1)

	### 05/22
	# Replace all of these classifiers by their CalibCV versions instead with 5 fold
	# We also want to throw in way more estimators into the mix with lots of random sizes to make it more robust
	# We can train different sub-sets based on different number of features say 38, and 43, 58 and then LoR them together
	# Hmm, one issue is that it wont work if we throw away points for the final LoR blending with different number of observations :(

	### Enumerate best performing classifiers for 38 feature test set
	"""
	clf_rfc1 = rfc(random_state = 30, criterion = 'entropy', max_depth = 6, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 40)
	clf_rfc2 = rfc(random_state = 30, criterion = 'gini', n_estimators = 80)	# wild-card
	clf_gbc = gbc(random_state = 30, max_depth = 3, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 80, subsample = 0.9, learning_rate = 0.05)
	clf_etc1 = etc(random_state = 30, n_estimators = 250, criterion = 'entropy', min_samples_split = 10)
	clf_etc2 = etc(random_state = 30, n_estimators = 200, criterion = 'gini')		# wild-card
	clf_adab = adab(random_state = 30, n_estimators = 350, learning_rate = 0.03)
	# pipe_adab = Pipeline([('adab', clf_adab), ('mmscaler', mmScaler)])	    # Does not work
	pipe_svc = Pipeline([('sscaler', scaler), ('svc', clf_svc)])
	"""

	### Enumerate best performing classifiers for 59 feature test set
	"""
	clf_rfc1 = rfc(random_state = 30, criterion = 'entropy', max_depth = 6, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 70, max_features = 7)
	clf_rfc2 = rfc(random_state = 30, criterion = 'gini', n_estimators = 100)	# wild-card
	clf_gbc = gbc(random_state = 30, max_depth = 3, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 120, subsample = 0.9, learning_rate = 0.03)
	clf_etc1 = etc(random_state = 30, n_estimators = 350, criterion = 'entropy', min_samples_leaf = 2, min_samples_split = 5)
	clf_etc2 = etc(random_state = 30, n_estimators = 250, criterion = 'gini')		# wild-card
	# clf_adab = adab(random_state = 30, n_estimators = 300, learning_rate = 0.02)
	"""
	
	### Enumerate best performing classifiers for 22 feature test set
	clf_rfc1 = rfc(random_state = 30, criterion = 'entropy', max_depth = 7, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 50)
	clf_rfc2 = rfc(random_state = 30, criterion = 'gini', n_estimators = 120, max_depth = 8, min_samples_split = 2, min_samples_leaf = 5)	
	clf_gbc = gbc(random_state = 30, max_depth = 4, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 80, subsample = 0.5, learning_rate = 0.03)
	clf_etc1 = etc(random_state = 30, n_estimators = 375, criterion = 'entropy', min_samples_leaf = 2, min_samples_split = 5)
	clf_etc2 = etc(random_state = 30, n_estimators = 60, criterion = 'gini',  min_samples_leaf = 2, min_samples_split = 10 )		
	
	clfs = [clf_rfc1,
					clf_rfc2,
					clf_etc1,
					clf_etc2,
					# pipe_adab,
					# clf_adab,
					# pipe_svc,
					clf_gbc]
	"""
	# Replace with Calibrated Classifiers instead	(cv5xccv5 is a bad idea because CCV splits are not stratified) - maybe try 3x3 or 5x2 instead 
	clfs = [CalibratedClassifierCV(clf_rfc1, cv=calib_folds, method='isotonic'),
					CalibratedClassifierCV(clf_gbc, cv=calib_folds, method='isotonic'),
					CalibratedClassifierCV(clf_etc1, cv=calib_folds, method='isotonic')]
	"""

	print ("Creating train and test sets for blending.")
    
	dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
	dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
	print ("dataset_blend_train.shape: ", dataset_blend_train.shape)
	print ("dataset_blend_test.shape: ", dataset_blend_test.shape)
    
	for j, clf in enumerate(clfs):
		print (j, clf)
		# This array will store num_folds columns of predictions 
		dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
		for i, (train, test) in enumerate(skf):
			print ("Fold", i)
			X_train = X[train]
			y_train = y[train]
			X_test = X[test]
			y_test = y[test]
			clf.fit(X_train, y_train)
			y_submission = clf.predict_proba(X_test)[:,1]
			# Fills in predictions one validation fold at a time 
			dataset_blend_train[test, j] = y_submission					
			dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
		# Prediction for the jth clf is a mean of the predictions over all folds for that clf and stored in jth column
		dataset_blend_test[:,j] = dataset_blend_test_j.mean(axis = 1)

	print ("Blending.")
	clf = LogisticRegression()
	clf.fit(dataset_blend_train, y)
	# Print weighting of classifiers
	print ("CLF weights after Logistic Regression: ", clf.intercept_, clf.coef_)

	y_submission = clf.predict_proba(dataset_blend_test)[:,1]
	# predict_prob = clf.predict_proba(dataset_blend_test)[:,1]

	# print ("Linear stretch of predictions to [0,1]")
	# y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
	y_submission = MinMaxScaler().fit_transform(y_submission)

	### Dump to csv
	print ("Saving Results.")
	# np.savetxt(fname='D:/Kaggle/HumanVRobot/results/temp_submit.csv', X=y_submission, fmt='%0.9f')
	
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/blend_38feat.csv', 'w', newline = ''))
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/blend_38feat_cv5_3p_mms.csv', 'w', newline = ''))
	# csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/blend_59feat_cv3_ccv3_3p_mms.csv', 'w', newline = ''))
	csv_out = csv.writer(open('D:/Kaggle/HumanVRobot/results/blend_21feat_rlr_cv5_5p.csv', 'w', newline = ''))
	data = []
	data.append(['bidder_id', 'prediction'])
	for idx in range(len(y_submission)):
		data.append([bidder_ids[idx], y_submission[idx]]) 					# We want to only dump class probabilities for bot

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
