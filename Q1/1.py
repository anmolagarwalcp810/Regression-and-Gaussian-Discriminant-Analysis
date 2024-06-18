import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-x',dest='x_csv',required=False,default='../ass1_data/data/q1/linearX.csv',help='Path to csv file linearX.csv')
parser.add_argument('-y',dest='y_csv',required=False,default='../ass1_data/data/q1/linearY.csv',help='Path to csv file linearY.csv')
parser.add_argument('-b',dest='b',required=False,default='1b.png',help='Path to png file for Q1.b')
parser.add_argument('-c1',dest='c1',required=False,default='1c.png',help='Path to png file for Q1.c')
parser.add_argument('-d1',dest='d1',required=False,default='1d.png',help='Path to png file for Q1.d')
parser.add_argument('-c2',dest='c2',required=False,default='1c.mp4',help='Path to mp4 file for Q1.c')
parser.add_argument('-d2',dest='d2',required=False,default='1d.mp4',help='Path to mp4 file for Q1.d')
parser.add_argument('-n',dest='n',required=False,default=0.025,type=float,help='learning rate')
parser.add_argument('--delta',required=False,default=1e-10,type=float,help='Convergence criteria')

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

def batch_gradient_descent(x,y,theta,lr=args.n,delta = args.delta):
    losses = []
    theta_0 = []
    theta_1 = []
    m = x.shape[1]
    # for k in range(iters):
    prev_loss = 100
    loss = 0
    k = 0
    while(True):# or k<iters):
        loss = np.sum(np.power(y-np.matmul(theta.T,x),2))/(2*m)
        losses.append(loss)
        theta_0.append(theta[0][0])
        theta_1.append(theta[1][0])
        if abs(prev_loss - loss) < delta: break
        else: prev_loss = loss
        # print("iter {}: {}".format(k,loss))
        theta = theta + lr*np.sum((y-(np.matmul(theta.T,x)))*x,axis=1,keepdims=True)/m
        k+=1

    return theta, losses, k, theta_0, theta_1

def J(theta_0,theta_1,x,y):
    m = x.shape[1]
    # return np.sum(np.power(y-np.matmul(theta.T,x),2)
    return np.sum(np.power(y-(theta_0*x[0]+theta_1*x[1]),2))/(2*m)

'''
Steps:
1. Read csv: Done
2. normalize to get mean = 0 and standard deviation = 1 
'''

'''
TODO: Get file path arguments (x_csv, y_csv)
'''

# x_csv = '../ass1_data/data/q1/linearX.csv'
# y_csv = '../ass1_data/data/q1/linearY.csv'

x_csv = args.x_csv
y_csv = args.y_csv

x_data = pd.read_csv(x_csv,header=None)
y_data = pd.read_csv(y_csv,header=None)

# print(x_data[0][0])
# print(y_data[0][0])

x_data = np.array(x_data[0].tolist())
y_data = np.array(y_data[0].tolist())

# print(x_data, len(x_data))
# print(y_data, len(y_data))

# normalize
x_data = normalize(x_data)
# print(x_data,len(x_data))
# print(np.mean(x_data),np.std(x_data))

# add x0
x = np.ones((2,x_data.shape[0]))
x[1,:] = x_data


# reshape y_data
y_data = y_data.reshape((1,y_data.shape[0]))

'''
Steps:
3. Batch Gradient Descent
'''

theta = np.zeros((2,1))
lr=args.n

# print(np.sum((y_data-(np.matmul(theta.T,x)))*x,axis=1,keepdims=True))
# checking broadcast
# print((y_data-(np.matmul(theta.T,x)))*x)

# print(theta.T)

theta, losses, iters, theta_0, theta_1 = batch_gradient_descent(x,y_data,theta,lr=lr)

print("theta:\n{}".format(theta))
print("learning rate: {}".format(lr))
print("Final Loss: {}".format(losses[-1]))
print("iters: {}".format(iters))

# b
fig0 = plt.figure(0)
plt.scatter(x[1],y_data,marker='.',label='Training Data')
plt.plot(x[1],np.matmul(theta.T,x)[0],c='g',label='Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.savefig(args.b)
# plt.show()


# c
fig1 = plt.figure(1)
ax = fig1.add_subplot(projection='3d')
m = x.shape[1]
theta_x = np.linspace(-1,3)
theta_y = np.linspace(-1,1)
theta_X, theta_Y = np.meshgrid(theta_x,theta_y)
z = np.zeros(theta_X.shape)
for i in range(theta_X.shape[0]):
    for j in range(theta_X.shape[1]):
        z[i][j] = J(theta_X[i][j],theta_Y[i][j],x,y_data)
ax.plot_surface(theta_X,theta_Y,z,cmap=cm.GnBu,alpha=0.5)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('J')
plt.title('Loss Function vs Parameters')

losses = np.array(losses)
theta_0 = np.array(theta_0)
theta_1 = np.array(theta_1)
line_gradient_descent, = ax.plot([],[],[],color='orange',label='Gradient Descent')
text_loss = ax.text(0,0,0,s='Loss: ',transform=ax.transAxes)
plt.legend()
def animate1(i,line,text):
    line.set_data(theta_0[:i],theta_1[:i])
    line.set_3d_properties(losses[:i])
    text.set_text('Loss: {}'.format(losses[i]))
    return line, text

animation1 = animation.FuncAnimation(fig1,func=animate1,fargs=(line_gradient_descent,text_loss),frames=len(losses),interval=200)
animation1.save(args.c2,writer='ffmpeg',fps=30)
plt.savefig(args.c1)
plt.show()

# d
fig2, ax2 = plt.subplots()
theta_x = np.linspace(-0.1,2)
theta_y = np.linspace(-0.0025,0.0025)
theta_X, theta_Y = np.meshgrid(theta_x,theta_y)
contour = plt.contourf(theta_X,theta_Y,z)
fig2.colorbar(contour)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.title('Loss Function vs Parameters')

# animation (theta_0, theta_1)
line_gradient_descent, = plt.plot([],[],color='orange',label='Gradient Descent')
text_loss = plt.text(0.01,0.01,s='Loss: ',transform=ax2.transAxes)
plt.legend()

def animate2(i,line,text):
    line.set_data(theta_0[:i],theta_1[:i])
    text.set_text('Loss: {}'.format(losses[i]))
    return line, text

animation2 = animation.FuncAnimation(fig2,func=animate2,fargs=(line_gradient_descent,text_loss),frames=len(theta_1),interval=200)

animation2.save(args.d2,writer='ffmpeg',fps=30)

plt.savefig(args.d1)
plt.show()