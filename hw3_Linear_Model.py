import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
from sklearn.metrics import mean_squared_error, r2_score

#genate Legendre polynomial
def Legendre_poly(x,k):
    if (0==k):
        return 1
    elif (1==k):
        return x
    else :
        r = (2*k-1)/k * x * Legendre_poly(x,k-1) - (k-1)/k * Legendre_poly(x,k-2)
        return r


'''
target function,
aq is coefficients
'''
def target_func(aq,x,noise):
    fx = 0
    degrees = len(aq)
    for n in range(degrees):
        fx = fx + aq[n] * Legendre_poly(x,n)
    fx = fx + noise
    return fx
    

def generate_date_set(degrees,sample,sigma,is_show,show_f):
    x_list = []
    for n in range(sample):
        x_list.append(random.uniform(-1,1))

    list.sort(x_list)

    aq = np.random.normal(0, 1, degrees+1)

    nos_c = np.random.normal(0, 1, sample)
    noise = nos_c*sigma
    if(show_f):
        y_list_pure = []
        for n in range(sample):
            y_list_pure.append(target_func(aq,x_list[n],0))

    y_list = []
    for n in range(sample):
        y_list.append(target_func(aq,x_list[n],noise[n]))
        
    if(is_show):
        plt.scatter(x_list,y_list,linewidths=1,alpha=0.5)
    if(show_f):
        plt.plot(x_list,y_list_pure,c='r')
    return (x_list,y_list)



def fit_data(x_list,y_list,N,is_show):

    # divide train and test data
    x_train = x_list[0:N]
    y_train = y_list[0:N]
    x_test = x_list[N:]
    y_test = y_list[N:]

    x_train_a = np.array(x_train)
    x_train_a = x_train_a.reshape(x_train_a.shape[0], 1)
    x_test_a = np.array(x_test)
    x_test_a = x_test_a.reshape(x_test_a.shape[0], 1)

    #plt.scatter(x_test,y_test,marker='x',c='r',linewidths=1,alpha=0.5)
    
    #fit g2
    polynomial_2 = PolynomialFeatures(2)
    x_order_2 = polynomial_2.fit_transform(x_train_a)

    poly_linear_model = LinearRegression()
    poly_linear_model.fit(x_order_2, y_train)

    
    x_order_test_2 = polynomial_2.fit_transform(x_test_a)
    y_pred_2 = poly_linear_model.predict(x_order_test_2)
    Eout_2 = mean_squared_error(y_test,y_pred_2)
    print("Eout_2: ",Eout_2)
    
    if (is_show):
        x_p = np.arange(-1,1,0.001)
        x_p_t = polynomial_2.fit_transform(x_p.reshape(x_p.shape[0], 1))
        y_p = poly_linear_model.predict(x_p_t)
        plt.plot(x_p, y_p,label = '$g2$')

        
    #fit g10
    polynomial_10 = PolynomialFeatures(10)
    x_order_10 = polynomial_10.fit_transform(x_train_a)

    poly_linear_model = LinearRegression()
    poly_linear_model.fit(x_order_10, y_train)

    x_order_test_10 = polynomial_10.fit_transform(x_test_a)
    y_pred_10 = poly_linear_model.predict(x_order_test_10)
    Eout_10 = mean_squared_error(y_test,y_pred_10)
    print("Eout_10: ",Eout_10)

    if (is_show):
        x_p = np.arange(-1,0.99,0.001)
        x_p_t_10 = polynomial_10.fit_transform(x_p.reshape(x_p.shape[0], 1))
        y_p_10 = poly_linear_model.predict(x_p_t_10)
        plt.plot(x_p, y_p_10,label = '$g10$')
        plt.legend()
        plt.show()

    
    return (Eout_2,Eout_10)


def test_generate_date_set():
    is_show = 1 # whether show the fit line 
    show_f = 1 # whether show the target function line
    Qf = 15
    N = 120
    sigma = 0.3
    sample = int(N+outN)
    generate_date_set(Qf,sample,sigma,is_show,show_f)
    plt.show()

    
def test_fit():
    is_show = 1 # whether show the fit line 
    show_f = 0 # whether show the target function line
    sigma = 0.1
    Qf = 15
    N = 100
    sample = int(N+outN)
    x,y = generate_date_set(Qf,sample,sigma,is_show,show_f)
    fit_data(x,y,N,1)

def test_overfit_N():
    is_show = 0 # whether show the fit line 
    show_f = 0 # whether show the target function line
    sigma = 0.1
    Qf = 15
    repeat = 10 #numbers of experiments that repeat for average
    E2,E10 = 0, 0
    Eout = []
    x_list = []
    for N in np.arange(20,121,5):
        sample = int(N+outN)
        for R in range(repeat):
            x,y = generate_date_set(Qf,sample,sigma,is_show,show_f)
            E2t,E10t = fit_data(x,y,N,is_show)
            E2 = E2+E2t
            E10 = E10+E10t
        E2 = E2/repeat
        E10 = E10/repeat
        print("E2:%f,E10:%f" %(E2,E10))
        x_list.append(N)
        Eout.append(E10-E2)

    plt.plot(x_list,Eout)
    print(Eout)
    plt.xlabel('number of samples, fixed Qf=15,sigma=0.1')
    plt.ylabel('square loss, E10-E2')
    plt.show()

def test_overfit_Qf():
    is_show = 0
    show_f = 0
    sigma = 0.1
    N = 100
    repeat = 10
    E2,E10 = 0, 0
    Eout = []
    x_list = []
    for Qf in np.arange(1,20,1):
        sample = int(N+outN)
        for R in range(repeat):
            x,y = generate_date_set(Qf,sample,sigma,is_show,show_f)
            E2t,E10t = fit_data(x,y,N,is_show)
            E2 = E2+E2t
            E10 = E10+E10t
        E2 = E2/repeat
        E10 = E10/repeat
        print("E2:%f,E10:%f" %(E2,E10))
        x_list.append(Qf)
        Eout.append(E10-E2)

    plt.plot(x_list,Eout)
    print(Eout)
    plt.xlabel('x=Qf , fixed N=100,sigma=0.1')
    plt.ylabel('square loss, E10-E2')
    plt.show()

def test_overfit_sigma2():
    is_show = 0
    show_f = 0
    N = 100
    Qf = 10
    repeat = 15
    E2,E10 = 0, 0
    Eout = []
    x_list = []
    for sigma2 in np.arange(0,2,0.05):
        sigma = np.sqrt(sigma2)
        sample = int(N+outN)
        for R in range(repeat):
            x,y = generate_date_set(Qf,sample,sigma,is_show,show_f)
            E2t,E10t = fit_data(x,y,N,is_show)
            E2 = E2+E2t
            E10 = E10+E10t
        E2 = E2/repeat
        E10 = E10/repeat
        print("E2:%f,E10:%f" %(E2,E10))
        x_list.append(sigma2)
        Eout.append(E10-E2)

    plt.plot(x_list,Eout)
    print(Eout)
    plt.xlabel('x=sigma^2 , fixed N=100,Qf=10')
    plt.ylabel('square loss, E10-E2')
    plt.show()





print("homework 3, cheng guo")
print("please input the number:")
print("1: run test_generate_date_set() that generate date set for test")
print("2: run test_fit() that fit date set for test")
print("3: run measure_overfit_N() that measure overfit by N")
print("4: run measure_overfit_Qf() that measure overfit by Qf")
print("5: run measure_overfit_sigma2() that measure overfit by sigma")
outN = 3 #number of test date

key = input()
if('1'==key):
    test_generate_date_set()
elif('2'==key):
    test_fit()
elif('3'==key):
    test_overfit_N()
elif('4'==key):
    test_overfit_Qf()
elif('5'==key):
    test_overfit_sigma2()
else:
    print("sorry, input beyond: ",key)



