import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# load the diabetes datasets
dataset = datasets.load_diabetes()
# prepare a range of alpha values to test
#alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
k_range = list(range(1, 31))
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=dict(n_neighbors=k_range))
grid.fit(dataset.data, dataset.target)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

