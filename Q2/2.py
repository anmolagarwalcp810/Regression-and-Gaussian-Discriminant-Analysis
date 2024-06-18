import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-t',dest='t',required=False,default='../ass1_data/data/q2/q2test.csv',help='Path to csv file q2test.csv')
parser.add_argument('-l',dest='l',required=False,default='2_losses.png',help='Path to png file showing loss vs iterations in SGD.')
parser.add_argument('-p',dest='p',required=False,default='2.png',help='Path to png file for Q2 showing theta')
parser.add_argument('-n',dest='n',required=False,default=0.001,type=float,help='learning rate')
parser.add_argument('-b',dest='b',required=False,default=1,type=int,help='batch size')
parser.add_argument('-k',dest='k',required=False,default=5000,type=int,help='k')
parser.add_argument('--delta',required=False,default=8e-5,type=float,help='Convergence criteria')

args = parser.parse_args()

'2.a'

size = int(1e6)

x = np.ones((3,size))

mu_x1, sigma_x1 = 3, 2
mu_x2, sigma_x2 = -1, 2

# square of standard deviation is called variance

x[1] = np.random.normal(mu_x1,sigma_x1,size)
x[2] = np.random.normal(mu_x2,sigma_x2,size)

# now we have x1 ~ N(3,4), x2 ~ N(-1,4)

theta = np.array([[3],[1],[2]])

y = np.zeros((size,1))

sigma_y = np.sqrt(2)

for i in range(size):
    # print(np.matmul(theta.T,x[:,i]))
    y[i][0] = np.random.normal(np.matmul(theta.T,x[:,i])[0],sigma_y)

# print(y)

'2.b'

# shuffle
permutations = np.random.permutation(size)
x = x[:,permutations]
# print(y.shape)
y = y[permutations]

theta1 = np.zeros((3,1))
lr = args.n
b = args.b
delta = args.delta
k = args.k

def loss(x,y,theta1,m):
    # print(theta1)
    return np.sum(np.power(y-np.matmul(theta1.T,x).T,2))/(2*m)

def stochastic_gradient_descent(x=x,y=y,theta1=theta1,b=b,lr=lr,k=k):
    prev, cur = -1,0
    i, a = 0, 0
    iters = 0
    loss_array = []
    theta_output = []
    while(True):
        # if iters%100==0:
        #     print(iters)
        #     print(prev,cur)
        theta_output.append([theta1[0][0],theta1[1][0],theta1[2][0]])
        if a == k:
            if(abs(prev-cur)<delta*k): break
            else:
                prev = cur
                cur = 0
                a = 0
        batch_x = x[:,i:i+b]
        batch_y = y[i:i+b]
        J = loss(batch_x,batch_y,theta1,b)
        # print(batch_y.shape)
        # print(batch_x.shape)
        # print(theta1.shape)
        # print(np.matmul(theta1.T,batch_x).T.shape)
        # print(np.sum((batch_y - (np.matmul(theta1.T, batch_x)).T).T * batch_x, axis=1,keepdims=True).shape)
        loss_array.append(J)
        theta1 = theta1 + lr * np.sum((batch_y - (np.matmul(theta1.T, batch_x)).T).T * batch_x, axis=1,keepdims=True)/b
        i=(i+b)%size
        a += 1
        # cur = (cur*(a-1) + J)/a
        cur = cur + J
        iters+=1

    return theta1, loss_array, iters, theta_output

theta1, losses, iters, theta_output = stochastic_gradient_descent(x,y,theta1,b,lr)

print("predicted theta: \n{}\niterations: {}\nfinal loss: {}".format(theta1,iters,losses[-1]))

fig0 = plt.figure(0)
plt.plot([i for i in range(iters)],losses)
plt.title('Loss vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig(args.l)
plt.show()

# x_csv = '../ass1_data/data/q2/q2test.csv'
x_csv = args.t

x_data = pd.read_csv(x_csv)

size1 = len(x_data['Y'])

y1 = np.zeros((size1,1))

y1[:,0] = np.array(x_data['Y'].to_list())
x1 = np.ones((3,size1))

x1[1] = np.array(x_data['X_1'].to_list())
x1[2] = np.array(x_data['X_2'].to_list())

print('Loss w.r.t Predicted Theta: {}\nLoss w.r.t Original Theta: {}\n'.format(loss(x1,y1,theta1,size1),loss(x1,y1,theta,size1)))

fig1 = plt.figure(1)
ax = fig1.add_subplot(projection='3d')
ax.plot([i[0] for i in theta_output],[i[1] for i in theta_output],[i[2] for i in theta_output])
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('theta_2')
plt.title('Movement of Theta with Iterations')
plt.savefig(args.p)
plt.show()
