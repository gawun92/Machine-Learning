import numpy as np
import matplotlib.pyplot as plt

######################### DATA 1 #####################################
fp =  np.genfromtxt('data1.csv', delimiter=',')
data1_x = []
data1_y = []

for index in fp:
    data1_x.append([index[0], index[1]])
    data1_y.append(index[2])
    if index[2] == 1:
        plt.scatter(index[0], index[1], color='red', marker = 'o')
    else:
        plt.scatter(index[0], index[1], color='blue', marker = '*')

data1_x = np.array(data1_x)
data1_y = np.array(data1_y)

w1 = np.zeros(2)
b1 = 0
u1 = 0
for _ in range(1000):
    for di in range(len(data1_x)):
        if data1_y[di] * (w1.dot(data1_x[di]) + b1) <= 0:
            w1[0] = w1[0] + data1_y[di] * data1_x[di][0]
            w1[1] = w1[1] + data1_y[di] * data1_x[di][1]
            b1 = b1 + data1_y[di]
            u1 += 1
#  ([w1,w2].T)*[x1,x2] + b = 0
#  w1*x1 + w2*x2 + b = 0      let w2 be 'y'
#  y = (-w1/w2)x - b/w2
#
slope = -w1[0]/w1[1]; constant = -b1/w1[1]
plt.plot([-0.4, 1.0],[(-0.4)*slope + constant, (1)*slope + constant])
plt.show()
######################### DATA 1 #####################################

######################### DATA 2 #####################################
fp =  np.genfromtxt('data2.csv', delimiter=',')
data2_x = []
data2_y = []

for index in fp:
    data2_x.append([index[0], index[1]])
    data2_y.append(index[2])
    if index[2] == 1:
        plt.scatter(index[0], index[1], color='red', marker = 'o')
    else:
        plt.scatter(index[0], index[1], color='blue', marker = '*')

data2_x = np.array(data2_x)
data2_y = np.array(data2_y)

w2 = np.zeros(2)
b2 = 0
u2 = 0
for _ in range(1000):
    for di in range(len(data2_x)):
        if data2_y[di] * (w2.dot(data2_x[di]) + b2) <= 0:
            w2[0] = w2[0] + data2_y[di] * data2_x[di][0]
            w2[1] = w2[1] + data2_y[di] * data2_x[di][1]
            b2 = b2 + data2_y[di]
            u2 += 1
#  ([w1,w2].T)*[x1,x2] + b = 0
#  w1*x1 + w2*x2 + b = 0      let w2 be 'y'
#  y = (-w1/w2)x - b/w2
#
slope = -w2[0]/w2[1]; constant = -b2/w2[1]
plt.plot([-0.5, 1.0],[(-0.5)*slope + constant, (1)*slope + constant])
plt.show()
######################### DATA 2 #####################################

######################### DATA 3 #####################################
fp =  np.genfromtxt('data3.csv', delimiter=',')
data3_x = []
data3_y = []

for index in fp:
    data3_x.append([index[0], index[1]])
    data3_y.append(index[2])
    if index[2] == 1:
        plt.scatter(index[0], index[1], color='red', marker = 'o')
    else:
        plt.scatter(index[0], index[1], color='blue', marker = '*')

data3_x = np.array(data3_x)
data3_y = np.array(data3_y)

w3 = np.zeros(2)
b3 = 0
u3 = 0
for _ in range(1000):
    for di in range(len(data3_x)):
        if data3_y[di] * (w3.dot(data3_x[di]) + b3) <= 0:
            w3[0] = w3[0] + data3_y[di] * data3_x[di][0]
            w3[1] = w3[1] + data3_y[di] * data3_x[di][1]
            b3 = b3 + data3_y[di]
            u3 += 1
#  ([w1,w2].T)*[x1,x2] + b = 0
#  w1*x1 + w2*x2 + b = 0      let w2 be 'y'
#  y = (-w1/w2)x - b/w2
#
slope = -w3[0]/w3[1]; constant = -b3/w3[1]
plt.plot([-0.5, 1.0],[(-0.5)*slope + constant, (1)*slope + constant])
plt.show()
######################### DATA 3 #####################################
