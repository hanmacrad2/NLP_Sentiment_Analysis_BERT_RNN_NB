import numpy as np
import matplotlib.pyplot as plt
import json_lines
import pandas as pd 

import pickle 
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold ,train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score, roc_curve, auc

def pickle_result(result, filename):
    """ Pickle the result and save to a file """

    pickle.dump(result, open('roc_'+filename+".p", "wb"))

FILE_TRAIN = '../data/train_formatted.csv'
FILE_TEST = '../data/test_formatted.csv'

train = pd.read_csv(FILE_TRAIN,
    encoding='utf-8')
test = pd.read_csv(FILE_TEST,
    encoding='utf-8')




print(train.head())
features_train = train.pop("text").to_numpy()
labels_train = train.pop("polarity").to_numpy()

features_test = test.pop("text").to_numpy()
labels_test = test.pop("polarity").to_numpy()

vect = CountVectorizer(binary=True)

Xtrain=vect.fit_transform(features_train)
ytrain=labels_train

Xtest=vect.transform(features_test)
ytest= labels_test



findC = False
if(findC):

    acc = []
    stdErr = []
    cis = [0.001,0.01,0.1,1,5,10,50,100,500]
    for ci in cis:
        print("> C value  %.1f" % ci)
        kf = KFold(n_splits=5)
        temp1 = []; temp2 = []; temp3 = []
        meanAccuracy = []
        for train, test in kf.split(Xtrain):
            model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=ci).fit(Xtrain[train],ytrain[train])
            model.fit(Xtrain[train], ytrain[train])
            ypred = model.predict(Xtrain[test])
            meanAccuracy.append(f1_score(ytrain[test], ypred))
        print("\tAccuracy = %f\n" % np.array(meanAccuracy).mean())
        acc.append(np.array(meanAccuracy).mean())
        stdErr.append(np.array(meanAccuracy).std())

    acc = np.array(acc)
    stdErr = np.array(stdErr)

    plt.errorbar(cis, acc, yerr=stdErr)
    plt.title("C cross-validation")
    plt.xlabel('C')
    plt.ylabel('F1 Score')
    plt.show()


Cvalue = 0.1
logisticModel = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=100000, C=Cvalue).fit(Xtrain,ytrain)


print(confusion_matrix(ytest,logisticModel.predict(Xtest)))
print(accuracy_score(ytest,logisticModel.predict(Xtest)))
print(f1_score(ytest,logisticModel.predict(Xtest)))

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
print(confusion_matrix(ytest,dummy.predict(Xtest)))
print(accuracy_score(ytest,dummy.predict(Xtest)))
print(f1_score(ytest,dummy.predict(Xtest)))


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(ytest,logisticModel.decision_function(Xtest))
plt.plot(fpr, tpr, color='green')
titlee_aocc = "LogisticRegression [AUC= " + str(round(auc(fpr, tpr),2)) +"]"

dy_scores = dummy.predict_proba(Xtest)
fpr, tpr, threshold = roc_curve(ytest, dy_scores[:, 1])
plt.plot(fpr, tpr, color='red',linestyle='--')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

plt.legend([titlee_aocc,"Baseline"], loc='lower right',)

plt.show()


print("")
print("")
print("")
print("Model Scores")
#print("Validation loss: {}".format(test_loss))
print("Validation Accuracy: {}".format(accuracy_score(logisticModel.predict(Xtest), ytest)))
print("Validation F1-Score: {}".format(f1_score(logisticModel.predict(Xtest), ytest,average='micro')))
print("Classification Report")
print(confusion_matrix(logisticModel.predict(Xtest), ytest))
print(classification_report(logisticModel.predict(Xtest), ytest))

fpr, tpr, thresholds = roc_curve(ytest, logisticModel.decision_function(Xtest))
my_auc = auc(fpr, tpr)
# If you want to save the results to a file uncomment lines belows
pickle_result(fpr, 'fpr')
pickle_result(tpr, 'tpr')
pickle_result(thresholds, 'thresholds')

print('AUC: %s' % my_auc)
print("")
print("")
print("")
print("Dummy Model Scores")
#print("Validation loss: {}".format(test_loss))
print("Validation Accuracy: {}".format(accuracy_score(dummy.predict(Xtest), ytest)))
print("Validation F1-Score: {}".format(f1_score(dummy.predict(Xtest), ytest,average='micro')))
print("Classification Report")
print(confusion_matrix(dummy.predict(Xtest), ytest))
print(classification_report(dummy.predict(Xtest), ytest))

dy_scores = dummy.predict_proba(Xtest)
fpr, tpr, threshold = roc_curve(ytest, dy_scores[:, 1])
my_auc = auc(fpr, tpr)
print('AUC: %s' % my_auc)

"""
pickle_result(fpr, 'fpr')
pickle_result(tpr, 'tpr')
pickle_result(thresholds, 'thresholds')
"""