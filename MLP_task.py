import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split



X, Y = datasets.load_svmlight_file("liver-disorders.txt")
#X = X.todense()

#get 100 sample, and divide it to 80 for train and 20 for test
X = X[0:100]
Y = Y[0:100]
n_split = 80
x_train = X[0:n_split]
y_train = Y[0:n_split]
x_test = X[n_split:]
y_test = Y[n_split:]

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


Ein_list = []
Eout_list = []
alpha_list = np.array([-1,1,2,3,4]) # the alpha exponentiation

for i, p_alpha in enumerate(alpha_list):
    p_alpha = math.pow(10,p_alpha) #exponentiation
    clf = MLPClassifier(alpha=p_alpha,max_iter=1000,random_state=1)
    clf.fit(x_train, y_train)

    #Calculate the train and test error value 
    y_tr_pre = clf.predict(x_train)
    y_test_pre = clf.predict(x_test)
    as_Ein = 1- accuracy_score(y_tr_pre, y_train)
    as_Eout = 1- accuracy_score(y_test_pre, y_test)
    Ein_list.append(as_Ein)
    Eout_list.append(as_Eout)

    print("alpha = %8f as_Ein = %f as_Eout=%f" %(p_alpha, as_Ein, as_Eout))

# Plot curves
plt.plot(alpha_list,Ein_list,color='blue',label = '$Ein$')
plt.plot(alpha_list,Eout_list,color='red',label = '$Eout$')
plt.xlabel('alpha = 1/lambda = 10 power x')
plt.ylabel('Error accuracy value')
plt.legend()
plt.show()




