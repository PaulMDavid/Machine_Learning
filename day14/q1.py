from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
iris=load_iris()
X=iris.data
y=iris.target
def fn(s):
 global X,y
 mlp=MLPClassifier(hidden_layer_sizes=100,activation=s)
 #print(mlp.coefs_)
 #coefs throws error__python 2 problem
 mlp.fit(X,y)
 print(mlp.predict([[3,5,4,2]]))
 p=mlp.predict(X)
 print(confusion_matrix(y,p))

def main():
 i=7
 while(i!=0):
  i=input('Enter 1-tanh,2-relu,0-exit')
  if(i==1):
   ss='tanh'
   fn(ss)
  elif(i==2):
   ss='relu'
   fn(ss)

main()
 


