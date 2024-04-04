import math
import numpy as np
import matplotlib.pyplot as plt
# X=np.array([2000, 2400, 1800, 3200, 2100, 2800, 2300, 1700, 3700, 2200])
# Y=np.array([325000, 425000, 275000, 540000, 310000, 470000, 385000, 295000, 625000, 365000])
# X = np.array([1.0, 2.0])   #features
# Y = np.array([300.0, 500.0])   #target value


X= np.array([12.0, 15.0, 20.0, 8.0, 10.0, 18.0, 16.0, 22.0, 24.0, 13.0, 9.0, 11.0, 17.0, 19.0])

# Corresponding example data for house prices (in USD)
Y= np.array([185, 240, 320, 125, 160, 280, 250, 350, 390, 200, 140, 170, 270, 300])
# max_x=np.max(X_t)
# max_y=np.max(Y_t)
# X=X_t/max_x
# Y=Y_t/max_y
# X=X/1000
# Y=Y/1000
f_wb=np.array([])
cf=np.array([])
m=len(X)
def fxn(w,b,X):
    fxn = np.array([])
    for i in range(m):
        fxn_val = (w * X[i]) + b
        fxn = np.append(fxn, fxn_val)
    return fxn
def cost(fxn,Y):
    sum=0
    for i in range(m):
        sum=sum+(fxn[i]-Y[i])**2
    J=sum/(2*m)
    return J
def grad(w,b,fx,X,Y,alpha,n):
    dj_dw=0
    dj_db=0
    J_history=[]
    p_history=[]
    w_history=[]
    b_history=[]
    
    for i in range(n):
        fuc=fxn(w,b,X)
        for j in range(m):
            dj_dw=dj_dw+((fuc[j]-Y[j])*(X[j]))
            dj_db=dj_db+((fuc[j]-Y[j]))
        dj_dw=dj_dw/m
        dj_db=dj_db/m
        b = b - (alpha * dj_db)                            
        w = w - (alpha * dj_dw)
        if i<100000:      # prevent resource exhaustion
            function = fxn(w, b, X)  # Compute the updated values of fxn 
            J_history.append( cost(function, Y))
            p_history.append([w,b])
            w_history.append([w])
            b_history.append([b])
            if i% math.ceil(n/10) == 0:
                print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history,w_history,b_history

w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 0.00001
# run gradient descent
w_final, b_final, J_hist, p_hist ,w,b = grad(w_init, b_init,fxn(w_init, b_init,X),X,Y,tmp_alpha,iterations)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})") 
fig, (a1, a2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
a1.plot(w,J_hist)
a1.set_xlabel('w')
a1.set_ylabel('J')
a2.plot(b,J_hist)
a2.set_xlabel('b')
a2.set_ylabel('J')

plt.grid()
plt.show()
 
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()
# print(w)
# print(b)
# print(J_hist)

f=fxn(w_final,b_final,X)

plt.scatter(X,Y)
plt.plot(X,f,c='b')
plt.show()


