import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


X_train, y_train = datasets.load_svmlight_file("cpusmall.txt")



cpu_time = []
sample_x = []
Ein_list = []
div_n = 64
train_size = np.size(y_train)
div_count = int(train_size/div_n)
print("div_count ",div_count)
for n in range(div_n):
    X_part = X_train[0:(n+1)*div_count]
    y_part = y_train[0:(n+1)*div_count]

    #print(np.size(X_part),np.size(y_part))
    start = time.time()
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_part, y_part)
    end = time.time()
    if 0 == n:
        print("n = ", n)
        print('Coefficients: \n', regr.coef_)
        coef_0 = regr.coef_
    elif div_n-1 == n:
        print("n = ", n)
        print('Coefficients: \n', regr.coef_)
        coef_n = regr.coef_
    
    y_pred = regr.predict(X_part)
    Ein = mean_squared_error(y_pred,y_part)
    #Ein  = np.mean((y_pred-y_part)**2))
    
    Ein_list.append(Ein)
    cpu_time.append((end-start)*1000)
    sample_x.append((n+1)*div_count)


#print("cpu time = ",cpu_time)
#print("Ein_list = ",Ein_list)

plt.figure()

plt.subplot(3,1,1)
plt.plot(sample_x,Ein_list)
plt.xlabel('number of samples')
plt.ylabel('square loss')

plt.subplot(3,1,2)
plt.plot(sample_x,cpu_time)
plt.xlabel('number of samples')
plt.ylabel('cpu_time: ms')

plt.subplot(3,1,3)
plt.plot(coef_0,color='red')
plt.scatter(range(len(coef_0)), coef_0, alpha=0.3,color='red',marker='o')
plt.plot(coef_n,color='blue')
plt.scatter(range(len(coef_n)), coef_n, alpha=0.3,color='blue',marker='x')
plt.xlabel('learned weights ')



plt.show()


