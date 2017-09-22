import numpy as np
from stable_sigmoid import *
def logistic_loss_gradient(w,b, X, Y, alpha=0):
    n = X.shape[0]
    d = 1
    z = X.dot(w)+b  
    h = phi(z)
    grad= X.T.dot(h-Y)/n
    grad = grad + alpha * w
    return grad

def logistic_loss(w, b ,X, Y, alpha=0):
    n = X.shape[0]
    d = 1
    z = X.dot(w) +b
    l= - (np.dot(log_phi(z),Y)+np.dot(np.ones(n)-Y,log_one_minus_phi(z)))/n
    l = l + 0.5*  alpha * (np.linalg.norm(w) ** 2)
    return l

def logistic_loss_hessian( w,b ,X, Y, alpha=0):
    n = X.shape[0]
    d = 1
    z= X.dot(w)+b
    q=phi(z)
    h= np.array(q*(1-phi(z)))
    H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n
    H = H + alpha * np.eye(d, d) 
    return H 



def logistic_loss_Hv(w,b, X, Y, v,alpha=0): 
    n = X.shape[0]
    d = 1
    _z=X.dot(w)
    _z = phi(-_z)
    d_binary = _z * (1 - _z)
    wa = d_binary * X.dot(v)
    Hv = X.T.dot(wa)/n
    out = Hv + alpha * v
    return out