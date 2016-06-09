import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation

''' Data Manipulation and One-hot encoding '''
print 'Loading Data'
data_train = pd.read_csv('original_train.csv', delimiter='\t', header=None)  # Load data into a numpy array
data_test = pd.read_csv('original_test.csv', delimiter='\t', header=None, )  # Load data into a numpy array

print 'Constructing data frame'
data_train.columns = ['Click', 'Weekday', 'Hour', 'Timestamp', 'Logtype', 'UserID', 'UserAgent', 'IP', 'Region', 'City', 'AdExchange', 'Domain', 'URL', 'AnonymousID', 'AdID', 'AdWidth', 'AdHeight', 'AdVis', 'AdFormat', 'AdFloor', 'CreativeID', 'KeyPage', 'Advertiser', 'Tags']
data_test.columns = ['Weekday', 'Hour', 'Timestamp', 'Logtype', 'UserID', 'UserAgent', 'IP', 'Region', 'City', 'AdExchange', 'Domain', 'URL', 'AnonymousID', 'AdID', 'AdWidth', 'AdHeight', 'AdVis', 'AdFormat', 'AdFloor', 'CreativeID', 'KeyPage', 'Advertiser', 'Tags']
print 'print Constructed data frame'

print 'Starting One hot encoding tags'
large_tag_set = data_train[['Tags']].append(data_test[['Tags']])
large_tag_set.columns = ['Tags']
large_tag_set = large_tag_set['Tags'].str.join(sep='').str.get_dummies(sep=',')
tag_train_set = large_tag_set.head(2847802)
tag_test_set = large_tag_set.tail(545421)
print 'Finished One hot encoding tags'

print 'Choose specific columns'
label_set = data_train[['Click']]
data_train = data_train[['Weekday', 'Hour', 'Region', 'City', 'AdExchange', 'AdWidth', 'AdHeight', 'AdVis', 'AdFormat', 'AdFloor', 'CreativeID', 'KeyPage']]
data_test = data_test[['Weekday', 'Hour', 'Region', 'City', 'AdExchange', 'AdWidth', 'AdHeight', 'AdVis', 'AdFormat', 'AdFloor', 'CreativeID', 'KeyPage']]
frames = [data_train, data_test]
large_set = pd.concat(frames)
print "Specific columns chosen"

print 'Starting one-hot encoding'
large_set = pd.get_dummies(large_set, columns=['Weekday', 'Hour', 'AdExchange', 'AdVis', 'AdFormat', 'CreativeID', 'KeyPage'])
print 'Finished one-hot encoding'

print 'Reconstructing Train/Test Sets'
data_train = large_set.head(2847802)
data_train = pd.DataFrame(pd.concat([data_train, tag_train_set], axis=1))
data_test = large_set.tail(545421)
data_test = pd.DataFrame(pd.concat([data_test, tag_test_set], axis=1))
print 'Finished Reconstructing Train/Test Sets'
print data_train.shape
print data_test.shape


print 'Started Computing train set labels'
label_set = np.sign(label_set['Click'])
label_set[label_set == -1] = 0
print 'Finished computing train set labels'

# fit estimator
print "start XGBClassifier"
n_samples = data_train.shape[0]
est=XGBClassifier(n_estimators=200, learning_rate=0.1, silent= False)

print "start fitting"
est.fit(data_train, label_set)
# predict class labels
probs = est.predict_proba(data_test)

print "cross validation start"
cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, random_state=0)
scores = cross_validation.cross_val_score(est, data_train, label_set, cv=cv)
mean = np.mean(probs[:, 1])
std = np.std(probs[:, 1])
print "Test predicted Mean:", mean
print "Test predicted STD:", std
df = pd.DataFrame(probs[:, 1])
df.columns = ["Prediction"]
df.index += 1
df.to_csv("output_prediction.csv", index_label="Id")

