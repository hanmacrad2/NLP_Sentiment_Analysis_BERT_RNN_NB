#Final Result evaluation

#Imports
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



#Load in auc/roc
def load_pickled_results():
    with open('./nb/roc_fpr.p', 'rb') as f:
        n_fpr = pickle.load(f)
        print (n_fpr)
    with open('./nb/roc_tpr.p', 'rb') as f:
        n_tpr = pickle.load(f)
        print (n_tpr)
    with open('./rnn/roc_fpr.p', 'rb') as f:
        r_fpr = pickle.load(f)
        print (r_fpr )
    with open('./rnn/roc_tpr.p', 'rb') as f:
        r_tpr = pickle.load(f)
        print (r_tpr)
    with open('./lr/roc_fpr.p', 'rb') as f:
        l_fpr = pickle.load(f)
        print (r_fpr )
    with open('./lr/roc_tpr.p', 'rb') as f:
        l_tpr = pickle.load(f)
        print (r_tpr)
    
    with open('./bert/roc_fpr.p', 'rb') as f:
        b_fpr = pickle.load(f)
        print (r_fpr)
    with open('./bert/roc_tpr.p', 'rb') as f:
        b_tpr = pickle.load(f)
        print (r_tpr)
    return n_fpr,n_tpr,r_fpr,r_tpr,l_fpr,l_tpr,b_fpr,b_tpr



n_fpr,n_tpr,r_fpr,r_tpr,l_fpr,l_tpr,b_fpr,b_tpr = load_pickled_results()


plt.plot(n_fpr,n_tpr, color='red',linestyle='--')
plt.plot(r_fpr,r_tpr, color='green',linestyle='--')
plt.plot(b_fpr,b_tpr, color='gray',linestyle='--')
plt.plot(l_fpr,l_tpr, color='blue',linestyle='--')
plt.plot([0,1],[0,1], color='#a5533a',linestyle='--')

plt.title('ROC curve of all models')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

plt.legend(["NB","RNN","BERT","T Baseline","MF Baseline"], loc='lower right',)

plt.show()

#Plot AUC/ROC
#plt.plot(fpr,tpr, label = 'Baseline model - male')
#print('AUC = {}'.format(auc(fpr, tpr)))