#!/usr/local/bin/python

import os
import json
import numpy as np
import random
from sklearn.grid_search import GridSearchCV
import sklearn.ensemble
from sklearn.metrics import confusion_matrix
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
#from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import *
#from sklearn.feature_extraction.text import HashingVectorizer

train_file = os.path.join(".","train.json","train.json")

with open(train_file) as data_file:
	train_data = json.loads(data_file.read())

#random.shuffle(train_data)

dev_test_data = [' '.join(d["ingredients"]) for d in train_data[:len(train_data)/3]]
#dev_test_data = [' '.join(d1.replace(" ","").replace("-","") for d1 in d["ingredients"]) for d in train_data[:len(train_data)/3]]
dev_test_label = [d["cuisine"] for d in train_data[:len(train_data)/3]]
dev_train_data = [' '.join(d["ingredients"]) for d in train_data[len(train_data)/3:]]
#dev_train_data = [' '.join(d1.replace(" ","").replace("-","") for d1 in d["ingredients"]) for d in train_data[len(train_data)/3:]]
dev_train_label = [d["cuisine"] for d in train_data[len(train_data)/3:]]

#print np.shape(dev_train_data)
#print np.shape(dev_test_data)

def text_preprocessor(s):
	return s.lower().replace("-", " ").replace("_", " ")
	
def emsemble_models(model_list, data):
	from collections import Counter
	pred = []
	for d in data:
		point_pred = []
		for model in model_list:
			point_pred.append(model.predict(d)[0])
		c = Counter(point_pred)
		pred.append(c.most_common(1)[0][0])
	return pred
	
def calcAccuracy(labels, pred_labels):
	correct = 0
	for label, pred in zip(labels, pred_labels):
		if label == pred: correct += 1
	return float(correct)/float(len(labels))
		
	
print "vectorizing text..."
vec = TfidfVectorizer(preprocessor=text_preprocessor, ngram_range=(1,2), max_df=0.5)
dev_train_vec = vec.fit_transform(dev_train_data)
dev_test_vec = vec.transform(dev_test_data)


print "training model1..."

#params = {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], 'penalty': ['l1']}
# for L2, large regularization (C=10) appears to be better
# for L1, best: C=5.0
#model = LogisticRegression()
#model = GridSearchCV(LogisticRegression(), param_grid=params, scoring='accuracy')

model = LogisticRegression(C=10.0, penalty="l2")
model.fit(dev_train_vec, dev_train_label)
#print "best parameters:"
#print str(model.best_params_)
print "calculating scores..."
print "accuracy: %.4f" % model.score(dev_test_vec, dev_test_label)

print "training model2..."
#alphas = {'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]}
#model2 = GridSearchCV(MultinomialNB(), param_grid=alphas)
model2 = MultinomialNB(alpha=0.01)
model2.fit(dev_train_vec, dev_train_label)
#print "best parameters:"
#print str(model2.best_params_)
print "calculating scores..."
print "accuracy: %.4f" % model2.score(dev_test_vec, dev_test_label)

print "training model3..."
model3 = LogisticRegression(C=5.0, penalty="l1")
model3.fit(dev_train_vec, dev_train_label)
#print "best parameters:"
#print str(model.best_params_)
print "calculating scores..."
print "accuracy: %.4f" % model3.score(dev_test_vec, dev_test_label)

print "ensembling models..."
predicted_results = emsemble_models([model, model2, model3], dev_test_vec)
print str(predicted_results[:5])
print "calculating scores..."
print "accuracy: %.4f" % calcAccuracy(dev_test_label, predicted_results)


	

