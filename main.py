import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

dir(iris)
iris.feature_names
iris.data

# create df with data and column names from iris
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# add target
df['target'] = iris.target
# check name of target categories
iris.target_names
# add target names columns
df['target_names'] = df.target.apply(lambda x: iris.target_names[x])

# create a df for each target
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

# plot length and width of two species
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'], color='blue', marker='+')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

# plot length and width of two species
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'], color='blue', marker='+')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

# create X df
X = df.drop(['target','target_names'], axis='columns')
# create y df
y = df.target

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# create SVC obj
model = SVC()
# fit model
model.fit(X_train, y_train)
# accuracy
model.score(X_test, y_test)

# increase C, which control soft margin cost function, ie the influence of each support vector
model = SVC(C=10)
# fit model
model.fit(X_train, y_train)
# accuracy
model.score(X_test, y_test)

# increase gamma, which leads to high bias and low variance
model = SVC(gamma=9)
# fit model
model.fit(X_train, y_train)
# accuracy
model.score(X_test, y_test)



