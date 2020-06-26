import numpy as np
import csv
import math
import matplotlib.pyplot as plt


def Optimize_thr(x1, x2, y, w, x1orx2, geqclass):
    if x1orx2 == 'x1':
        error_opt = 10000
        for thr in range(math.floor(min(x1)), math.ceil(max(x1)) + 1):
            if geqclass == 1:
                y_predict = np.sign(x1 - thr)
            else:
                y_predict = np.sign(thr - x1)
            error_weighted = sum(np.multiply(w, (y_predict != y)))
            if error_weighted < error_opt:
                error_opt = error_weighted
                thr_opt = thr
    else:
        error_opt = 10000
        for thr in range(math.floor(min(x2)), math.ceil(max(x2)) + 1):
            if geqclass == 1:
                y_predict = np.sign(x2 - thr)
            else:
                y_predict = np.sign(thr - x2)
            error_weighted = sum(np.multiply(w, (y_predict != y)))
            if error_weighted < error_opt:
                error_opt = error_weighted
                thr_opt = thr
    return error_opt, thr_opt



f = open("AdaBoost_data.csv", 'r')
X = np.array(list(csv.reader(f, delimiter=','))).astype(float)

x1 = X[:,[0]]; x2 = X[:,[1]]; y = X[:,[2]]
w_mat = np.zeros((y.shape[0],3))
w_mat = np.zeros((y.shape[0],3))
w_mat[:,[0]]=np.divide(np.ones((y.shape[0],1)),y.shape[0])
e1 = np.array([[1,0,0]])
e3 = np.array([[0,0,1]])


error_opt1,thr_opt1 = Optimize_thr(x1,x2,y,w_mat[:,[0]],'x1',-1)
plt.scatter(x1,x2,s = np.multiply(w_mat[:,0],1000),c = np.dot((y==-1),e3)+np.dot((y==1),e1))
plt.plot([3,3],[0,7])
plt.title('1st Plot')
plt.show()

y_predict = np.sign(np.tile(thr_opt1,[10,1])-x1);
a1 = 0.5*math.log((1-error_opt1)/error_opt1);
w_mat[:,[1]] = np.multiply(w_mat[:,[0]],np.exp(-a1*np.multiply(y,y_predict)))
w_mat[:,1] = np.divide(w_mat[:,1],sum(w_mat[:,1]))

error_opt2,thr_opt2 = Optimize_thr(x1,x2,y,w_mat[:,[1]],'x1',-1)
plt.scatter(x1,x2,s = np.multiply(w_mat[:,1],1000),c = np.dot((y==-1),e3)+np.dot((y==1),e1))
plt.plot([7,7],[0,7])
plt.title('2nd Plot')
plt.show()


y_predict = np.sign(np.tile(thr_opt2,[10,1])-x1);
a2 = 0.5*math.log((1-error_opt2)/error_opt2);
w_mat[:,[2]] = np.multiply(w_mat[:,[1]],np.exp(-a2*np.multiply(y,y_predict)))
w_mat[:,2] = np.divide(w_mat[:,2],sum(w_mat[:,2]))
plt.scatter(x1,x2,s = np.multiply(w_mat[:,2],1000),c = np.dot((y==-1),e3)+np.dot((y==1),e1))
plt.plot([0,9],[5,5])
plt.title('3rd Plot')
plt.show()

error_opt3,thr_opt3 = Optimize_thr(x1,x2,y,w_mat[:,[2]],'x2',1)
y_predict = np.sign(x2-np.tile(thr_opt2,[10,1]))
a3 = 1/2*math.log((1-error_opt3)/error_opt3)

y_predict_final = np.sign(a1*np.sign(np.tile(thr_opt1,[10,1])-x1)
                          +a2*np.sign(np.tile(thr_opt2,[10,1])-x1)
                          +0.92*np.sign(x2-np.tile(thr_opt3,[10,1])))
print(y_predict_final)