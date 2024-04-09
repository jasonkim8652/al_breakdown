import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

title = "PGR"

df=pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_inf_ucb_substruct.csv")


x=df[["Aroma","HBA","HBD","Rotatable_bond","Ter_Amine","Sec_Amine","Pri_Amine","Ketone","Ester","Urea","Molecular Weight","Ring","Amide"]]
y=df[['Dock']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
'''
regr=MLPRegressor(random_state=1,max_iter=500).fit(x_train,y_train)
print(regr.score(x_train,y_train))

print(regr.score(x_test,y_test))



'''
mlr=LinearRegression()
mlr.fit(x_train,y_train)

print(mlr.coef_)
print(mlr.score(x_test,y_test))

pr=LinearRegression()
quadratic = PolynomialFeatures(degree=2)
x_quad = quadratic.fit_transform(x_train)

pr.fit(x_quad,y_train)

print(pr.coef_)
x_quad = quadratic.fit_transform(x_test)
print(pr.score(x_quad,y_test))

pr=LinearRegression()
quadratic = PolynomialFeatures(degree=3)
x_quad = quadratic.fit_transform(x)

pr.fit(x_quad,y)
print(pr.coef_)
print(pr.score(x_quad,y))

pr=LinearRegression()
quadratic = PolynomialFeatures(degree=4)
x_quad = quadratic.fit_transform(x_train)

pr.fit(x_quad,y_train)

print(pr.score(x_quad,y_train))

pr=LinearRegression()
quadratic = PolynomialFeatures(degree=5)
x_quad = quadratic.fit_transform(x_train)

pr.fit(x_quad,y_train)

print(pr.score(x_quad,y_train))

