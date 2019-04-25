import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


X, Y = datasets.load_svmlight_file("breast-cancer_scale.txt")

p_C_list = np.array([-3,-2,-1,1,2,3,4,5,6])
p_gamma_list = np.array([-7,-6,-5,-4,-3,-2,-1,1,2,3])
scores_list = []
opt_vaule = 0;
best_C = 0
best_gamma = 0;
z_vaule = np.zeros((len(p_C_list), len(p_gamma_list)))
for i, p_C in enumerate(p_C_list):
    p_C = math.pow(10,p_C)
    for j, p_gamma in enumerate(p_gamma_list):
        p_gamma = math.pow(10,p_gamma)
        clf = svm.SVC(kernel='rbf', C= p_C,gamma=p_gamma)
        scores = cross_val_score(clf, X, Y,scoring='accuracy',cv=4)
        mean_socr = scores.mean()
        z_vaule[i,j] = mean_socr
        print("mean_socr= %f,p_C=%8f p_gamma=%8f " %(mean_socr,p_C,p_gamma))
        if(mean_socr > opt_vaule):
            best_C = p_C
            best_gamma = p_gamma
            opt_vaule = mean_socr

print("the chosen C is %f,the chosen gamma=%f" %(best_C, best_gamma))


#train on the entire dataset
clf = svm.SVC(kernel='rbf', C= best_C,gamma=best_gamma)
clf.fit(X, Y)
y_predic = clf.predict(X)
a_s = accuracy_score(y_predic, Y)
print("use chosen C and gamma on the entire dataset, the accuracy vaule is  = ",a_s)


fig, ax = plt.subplots()
cs = ax.contourf(p_gamma_list,p_C_list, z_vaule, cmap=plt.cm.hot)
cbar = fig.colorbar(cs)
plt.xlabel('gamma = 10 power x')
plt.ylabel('C = 10 power y')
plt.show()


        


