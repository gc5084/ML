import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


X_train, y_train = datasets.load_svmlight_file("german.numer.txt")



cpu_time = []
sample_x = []
mean_list = []
div_n = 16
train_size = np.size(y_train)
div_count = int(train_size/div_n)
print("train_size= %d,div_n,=%d div_count=%d," %(train_size,div_n,div_count))
for n in range(div_n):
    X_part = X_train[0:(n+1)*div_count]
    y_part = y_train[0:(n+1)*div_count]

    #print(np.size(X_part),np.size(y_part))
    start = time.time()
    # Create logistic regression object
    clf = linear_model.LogisticRegression(solver='liblinear')
    # Train the model using the training sets
    clf.fit(X_part, y_part)
    end = time.time()
    y_pred = clf.predict(X_part)
    y_pred_p = clf.predict_proba(X_part)
    mean = clf.score(X_part, y_part)
    mean_list.append(mean)
    cpu_time.append((end-start)*1000)
    sample_x.append((n+1)*div_count)
    
    if div_n/2 == n:
        print("n = ", n)
        print('1Coefficients: \n', clf.coef_)
        coef_0 = clf.coef_
        print('Coefficients1111: \n', coef_0)

    elif div_n-1 == n:
        print("n = ", n)
        print('Coefficients: \n', clf.coef_)
        coef_n = clf.coef_



#print("cpu time = ",cpu_time)

plt.figure()

plt.subplot(3,1,1)
plt.plot(sample_x,mean_list)
plt.xlabel('number of samples')
plt.ylabel('mean accuracy')

plt.subplot(3,1,2)
plt.plot(sample_x,cpu_time)
plt.xlabel('number of samples')
plt.ylabel('cpu_time: ms')

plt.subplot(3,1,3)
plt.plot(coef_0[0],color='red')
plt.scatter(range(len(coef_0[0])), coef_0[0], alpha=0.3,color='red',marker='o') 
plt.plot(coef_n[0],color='blue')
plt.scatter(range(len(coef_n[0])), coef_n[0], alpha=0.3,color='blue',marker='x') 
plt.xlabel('learned weights ')


plt.show()


