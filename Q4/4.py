import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-x',dest='x_csv',required=False,default='../ass1_data/data/q4/q4x.dat',help='Path to csv file q4x.dat')
parser.add_argument('-y',dest='y_csv',required=False,default='../ass1_data/data/q4/q4y.dat',help='Path to csv file q4y.dat')
parser.add_argument('-p',dest='p',required=False,default='4.png',help='Path to png file for Q4 showing GDA decision boundaries')

args = parser.parse_args()

def normalize(x):
    '''
    :param x: linear X (numpy array)
    :return: normalized numpy arrays x
    '''
    mean = 0
    n = x.shape[0]
    for i in range(n):
        mean+=x[i]
    mean = mean/n
    variation = 0
    for i in range(n):
        variation += np.power(x[i]-mean,2)
    variation = variation/n
    x = (x-mean)/np.sqrt(variation)
    return x

def mu0(x,y):
    return (np.sum((1-y)*x,axis=1)/np.sum(1-y)).reshape((2,1))

def mu1(x,y):
    return (np.sum(y*x,axis=1)/np.sum(y)).reshape((2,1))

def sigma(x,y,mu,n,m):
    s = np.zeros((n,n))
    for i in range(m):
        s += np.matmul(x[:,i].reshape(2,1)-mu[y[0,i]],(x[:,i].reshape(2,1)-mu[y[0,i]]).T)
    s = s/m
    return s

def sigma0(x,y,mu_0,n,m):
    s = np.zeros((n,n))
    for i in range(m):
        if(y[0,i]==0):
            s += np.matmul(x[:,i].reshape(2,1)-mu_0,(x[:,i].reshape(2,1)-mu_0).T)
    s = s/np.sum(1-y)
    return s

def sigma1(x,y,mu_1,n,m):
    s = np.zeros((n,n))
    for i in range(m):
        if(y[0,i]==1):
            s += np.matmul(x[:,i].reshape(2,1)-mu_1,(x[:,i].reshape(2,1)-mu_1).T)
    s = s/np.sum(y)
    return s

def linear_decision_boundary(x1,mu_0,mu_1,sigma_01,c):
    sigma_inv = np.linalg.inv(sigma_01)
    term1 = (np.matmul(mu_0.T,sigma_inv) - np.matmul(mu_1.T,sigma_inv)).squeeze(0)
    term2 = np.matmul(np.matmul(mu_0.T,sigma_inv),mu_0) - np.matmul(np.matmul(mu_1.T,sigma_inv),mu_1)
    return (-term1[0]*x1 -c + term2[0]/2)/term1[1]

def quadratic_decision_boundary(x1,mu_0,mu_1,sigma_0,sigma_1,constant):
    sigma_0_inv = np.linalg.inv(sigma_0)
    sigma_1_inv = np.linalg.inv(sigma_1)
    t1 = sigma_0_inv-sigma_1_inv
    a = t1[0][0]
    b = t1[1][1]
    c = (t1[0][1]+t1[1][0])
    t2 = -2*(np.matmul(mu_0.T,sigma_0_inv)-np.matmul(mu_1.T,sigma_1_inv)).squeeze()
    d = t2[0]
    e = t2[1]
    f = -2*constant + np.matmul(mu_0.T,np.matmul(sigma_0_inv,mu_0)) - np.matmul(mu_1.T,np.matmul(sigma_1_inv,mu_1))
    x2 = (-(c*x1+e) +np.sqrt(np.square(c*x1+e)-4*b*(a*np.square(x1)+d*x1+f)))/(2*b)
    return x2


def quadratic_decision_boundary2(x1, mu_0, mu_1, sigma_0, sigma_1, constant):
    sigma_0_inv = np.linalg.inv(sigma_0)
    sigma_1_inv = np.linalg.inv(sigma_1)
    t1 = sigma_0_inv - sigma_1_inv
    a = t1[0][0]
    b = t1[1][1]
    c = (t1[0][1] + t1[1][0])
    t2 = -2 * (np.matmul(mu_0.T, sigma_0_inv) - np.matmul(mu_1.T, sigma_1_inv)).squeeze()
    # print(t2.shape)
    d = t2[0]
    e = t2[1]
    f = -2 * constant + np.matmul(mu_0.T, np.matmul(sigma_0_inv, mu_0)) - np.matmul(mu_1.T,
                                                                                    np.matmul(sigma_1_inv, mu_1))
    x2 = (-(c * x1 + e) - np.sqrt(np.square(c * x1 + e) - 4 * b * (a * np.square(x1) + d * x1 + f))) / (2 * b)
    return x2

# x_csv = '../ass1_data/data/q4/q4x.dat'
# y_csv = '../ass1_data/data/q4/q4y.dat'

x_csv = args.x_csv
y_csv = args.y_csv

x_data = pd.read_csv(x_csv,header=None,delimiter='  ',engine='python')
y_data = pd.read_csv(y_csv,header=None)

# print(x_data)
# print(y_data)

m = len(y_data)

x1 = np.array(x_data[0].tolist()).astype(dtype=float)
x2 = np.array(x_data[1].tolist()).astype(dtype=float)

x = np.zeros((2,m))

x[0] = normalize(x1)
x[1] = normalize(x2)

y = np.array((y_data[0]=='Canada').tolist()).astype(dtype=int).reshape((1,m))

n = 2

mu_0 = mu0(x,y)
mu_1 = mu1(x,y)
sigma_01 = sigma(x,y,[mu_0,mu_1],2,m)
sigma_0 = sigma0(x,y,mu_0,2,m)
sigma_1 = sigma1(x,y,mu_1,2,m)

print("mu_0 : \n{}\n\n mu_1 : \n{}\n\n sigma : \n{}\n\n sigma_0 : \n{}\n\n sigma_1 : \n{}\n\n".format(mu_0,mu_1,sigma_01,sigma_0,sigma_1))

y0 = np.where(y[0]==0)
y1 = np.where(y[0]==1)

phi = np.sum(y[0])/m
print('phi : {}'.format(phi))

# calculating linear decision boundary
x1 = np.linspace(-2,2.5)
c = np.log((1-phi)/phi)
x2 = linear_decision_boundary(x1,mu_0,mu_1,sigma_01,c)

# calculating quadratic decision boundary
c = np.log(((1-phi)/phi)*(np.sqrt(np.linalg.det(sigma_1))/np.sqrt(np.linalg.det(sigma_0))))
x2_quadratic = quadratic_decision_boundary2(x1,mu_0,mu_1,sigma_0,sigma_1,c).squeeze()

fig0 = plt.figure(0)
plt.scatter(x[0][y0],x[1][y0],marker='o',label='Alaska')
plt.scatter(x[0][y1],x[1][y1],marker='.',label='Canada')
plt.title('Salmons in Alaska vs Canada')
plt.xlabel('Fresh Water')
plt.ylabel('Marine Water')
plt.plot(x1,x2,color='orange',label='linear decision boundary')
plt.plot(x1,x2_quadratic,color='black',label='quadratic decision boundary')
plt.legend()
plt.savefig(args.p)
plt.show()