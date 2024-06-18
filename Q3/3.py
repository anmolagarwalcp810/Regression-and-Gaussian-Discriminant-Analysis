import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-x',dest='x_csv',required=False,default='../ass1_data/data/q3/logisticX.csv',help='Path to csv file linearX.csv')
parser.add_argument('-y',dest='y_csv',required=False,default='../ass1_data/data/q3/logisticY.csv',help='Path to csv file linearY.csv')
parser.add_argument('-p',dest='p',required=False,default='3.png',help='Path to png file for Q3 showing logistic regression')
parser.add_argument('--delta',required=False,default=1e-7,type=float,help='Convergence criteria')

args = parser.parse_args()


def normalize(x):
    '''
    :param x: linear X (numpy array)
    :return: normalized numpy arrays x
    '''
    mean = np.mean(x)
    standard_deviation = np.std(x)
    x = (x-mean)/standard_deviation
    return x

def sigmoid(z):
    return 1/(1+np.exp(-z))

def hessian(x,theta):
    n = theta.shape[0]
    h = np.zeros((n,n))
    z = np.matmul(theta.T,x)   # 1 x m
    a = sigmoid(z)*(1-sigmoid(z))   # 1 x m
    for k in range(n):
        for j in range(n):
            h[k][j] = -np.sum(np.matmul(a,(x[k]*x[j]).T))       # x[k] : 1 x m
    return h

def newton(x,y,theta,m,delta=args.delta):
    loss_array = []
    prev_loss = -1
    loss = -2
    while(abs(loss-prev_loss)>delta):
        # np.matmul(theta.T,x) : 1 x m
        # loss = np.sum(np.power(y-sigmoid(np.matmul(theta.T,x)),2))/(2*m)
        if loss!=-2:
            prev_loss = loss
        loss = np.sum(y*np.log(sigmoid(np.matmul(theta.T,x))) + (1-y)*np.log(1-sigmoid(np.matmul(theta.T,x))))/(2*m)
        loss_array.append(loss)
        # print(np.all(np.abs(hessian(x,theta)-hessian(x,theta).T)<1e-8))
        # print(np.all(np.linalg.eigvals(hessian(x,theta))<=0))
        # np.linalg.cholesky(-hessian(x,theta))
        # print('h^-1:\n{}\ndeltheta:\n{}\n'.format(np.linalg.inv(hessian(x,theta)),np.sum((y-sigmoid(np.matmul(theta.T,x)))*x,axis=1,keepdims=True)))
        theta = theta - np.matmul(np.linalg.inv(hessian(x,theta)),
                                  np.sum((y-sigmoid(np.matmul(theta.T,x)))*x,axis=1,keepdims=True))
        # iters-=1

    return theta, loss_array

# x_csv = '../ass1_data/data/q3/logisticX.csv'
# y_csv = '../ass1_data/data/q3/logisticY.csv'

x_csv = args.x_csv
y_csv = args.y_csv

x_data = pd.read_csv(x_csv,header=None)
y_data = pd.read_csv(y_csv,header=None)

x1 = np.array(x_data[0].tolist())
x2 = np.array(x_data[1].to_list())
y = np.array(y_data[0].tolist())

x1 = normalize(x1)      # individually normalize along each component
x2 = normalize(x2)

m = x1.shape[0]

x = np.ones((3,m))
x[1] = x1
x[2] = x2

y = y.reshape((1,m))

theta = np.zeros((3,1))

theta, losses = newton(x,y,theta,m)

print("theta:\n{}\nloss_array:\n{}\n".format(theta,losses))

# 3.b
y0 = np.where(y[0]==0)
y1 = np.where(y[0]==1)

# need to find x1, x2 where hypothesis is 0.5, thetaT x = 0
x_1 = np.arange(-3,4)
x_2 = -(theta[1][0]/theta[2][0])*x_1 - theta[0][0]/theta[2][0]

fig0 = plt.figure(0)
plt.scatter(x[1][y0],x[2][y0],marker='o',label='y=0')
plt.scatter(x[1][y1],x[2][y1],marker='.',label='y=1')
plt.plot(x_1,x_2,label='Decision Boundary',color='black')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Logistic Regression')
plt.legend()
plt.savefig(args.p)
plt.show()