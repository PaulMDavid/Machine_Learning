
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score,KFold
import matplotlib.pyplot as plt

import pandas as pd

def eval_classifiers(X,y,data_name,ignore_svm=False):
    plt.close()
    models = [("KNN",KNeighborsClassifier()), ("NB",GaussianNB()), ("LR",LogisticRegression()), ("CART",DecisionTreeClassifier())]
    if(not ignore_svm):  
 #SVM takes too long
        models.append(("SVM",SVC()))
        
    models.sort()
    results = []
    names = []
    scoring = 'accuracy'
    
    for name,model in models:
        kfold = KFold(n_splits = 10, random_state = 7)
        score = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
        results.append(score)
        names.append(name)
    fig = plt.figure()
    fig.suptitle('Classifier Comparision on %s'%(data_name))
    ax=fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
from sklearn.datasets import load_iris

iris = load_iris() 
X = iris.data
y = iris.target

eval_classifiers(X,y,'Iris')
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
data=pd.read_csv(url)
data=data.as_matrix()
X=data[:,0:8]
y=data[:,8]
eval_classifiers(X,y,'Pima Indians')
