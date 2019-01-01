import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv('/home/ai11/Desktop/common/ML/Day2/Questions/Immunotherapy.csv')
age = data['age']
data = data.as_matrix()
#type(data) - numpy.ndarray
X = data[:,0:6]
X = preprocessing.scale(X)
X_train,X_test,y_train,y_test = train_test_split(X,data[:,7]) 

accu_scores = []

for i in range(1,20):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    p = model.predict(X_test)
    accu_scores.append(accuracy_score(y_test,p))

print accu_scores   
plt.plot(range(1,20),accu_scores)
plt.xlabel("K values")
plt.ylabel("Accuracy Scores")
plt.show()
