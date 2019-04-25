from numpy import *
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def createSeparatePoint(weights,pointNums):
    ar_w =array(weights)
    FeatureNums = len(weights)
    pointDataSet = zeros((pointNums,FeatureNums+1))

    for i in range(pointNums):
        x = random.rand(1,FeatureNums)*20 - 10
        leftOrRight = sum(ar_w*x)
        if leftOrRight <= 0:
            pointDataSet[i] = append(x,-1)
        else:
            pointDataSet[i] = append(x,1)

    return pointDataSet


def plotPoint(dataSet):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('random point set')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PLA')
    labels = array(dataSet[:,2])
    print(labels)
    idx_true = where(dataSet[:,2]==1)  #get true date
    print(idx_true)
    p1 = ax.scatter(dataSet[idx_true,0],dataSet[idx_true,1],marker='o',c='g',label='true')
    idx_false = where(dataSet[:,2]==-1) #get false date
    print(idx_false)
    p2 = ax.scatter(dataSet[idx_false,0],dataSet[idx_false,1],marker='x',c='r',label='false')
    plt.legend(loc = 'upper right')
    input('press any key to continue')
    plt.show()
    return



def PLA_process(weights,dataSet):
    print("Enter PLA_process")
    pointNum = dataSet.shape[0]
    FeatureNum = dataSet.shape[1]
    w = zeros((1,FeatureNum-1))

    n=0
    updates = 0
    while n<pointNum:
        if dataSet[n][-1] * sum(w*dataSet[n,0:-1]) <= 0:
            w = w + dataSet[n][-1]*dataSet[n,0:-1]
            n = 0
            updates += 1
        else:
            n += 1

    print("updates = ", updates)
    dims = len(weights)
    print("dimension = ",dims)
    print("hypothesis g", w)
    
    if len(weights) < 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('PLA, yellow: target f, blue: hypothesis')
        plt.xlabel('X')
        plt.ylabel('Y')
        labels = array(dataSet[:,2])
        #print(labels)
        idx_true = where(dataSet[:,2]==1)  #get true date
        #print(idx_true)
        p1 = ax.scatter(dataSet[idx_true,0],dataSet[idx_true,1],marker='o',c='g',label='true')
        idx_false = where(dataSet[:,2]==-1) #get false date
        #print(idx_false)
        p2 = ax.scatter(dataSet[idx_false,0],dataSet[idx_false,1],marker='x',c='r',label='false')
        ys = (-12 * (-w[0][0]) / w[0][1], 12 * (-w[0][0]) / w[0][1])
        ###set the two points x to -12,12, then get the y value
        ax.add_line(Line2D((-12, 12), ys, linewidth=1, color='blue'))

        ###add ideal f line
        fy = (-12* (-weights[0])/weights[1], 12*(-weights[0])/weights[1] )
        ax.add_line(Line2D((-12, 12), fy, linewidth=1, color='yellow'))
        plt.legend(loc = 'upper right')
        plt.show()
    
    return


    
print('start: \n')
wht = [3,-2]  #weights,  target function
#wht = [3,-2,3,4,5,6,7,8,9,10]   
dset = createSeparatePoint(wht,20) #parameter 2 is the number of point
print(dset)
input('press any key to continue')
#plotPoint(aa)
PLA_process(wht,dset)








































