import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score

# Load data
X, Y = datasets.load_svmlight_file("fourclass.txt")
num_s = X.shape[0]
for n in range(4):
    div = int(num_s/5*(n+1))
    
    x_train = X[:div]
    y_train = Y[:div]
    x_test = X[div:]
    y_test = Y[div:]

    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(x_train, y_train)

    y_pre = clf.predict(x_test)

    #count the error number
    err_accuracy = 1- accuracy_score(y_pre, y_test)
    print("the classification error precent is ",err_accuracy)


    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("decison_task"+str(n))
    
