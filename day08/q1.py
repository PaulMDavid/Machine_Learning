import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

house_data = load_boston()
X = house_data.data
y = house_data.target
rmses = []
def test_model(m_name,model):
    global X,y,rmses
    model.fit(X,y)
    p = model.predict(X)
    rmses.append((m_name,(mean_squared_error(y,p) ** 0.5)))
    print "RMSE for ",m_name," : ",(mean_squared_error(y,p) ** 0.5)
    plt.scatter(y,p)
    plt.title(m_name)
    plt.show()
rmses = []
test_model("Linear _ LASSO",Lasso())
test_model("SVR",SVR())
test_model("RandomForestRegressor",RandomForestRegressor())
test_model("KNeighborsRegressor",KNeighborsRegressor())
test_model("Ridge",Ridge())
print rmses


