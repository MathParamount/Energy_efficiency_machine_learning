from pre_training import *
from src.polinomial_function import *

def cost_huber(x_poly,y,w,b,delta= 1.0):
    m = len(y)
    y_pred = x_poly.dot(w) + b      #Prediction of target
    erro = y_pred - y

    #Huber loss calculation - Vetorization
    mask = np.abs(erro) <= delta
    loss = np.where(mask, (1/2)*erro**2, delta * np.abs(erro) - (1/2)*delta**2)
    
    return np.sum(loss)/m

def gradient_huber(x_poly,y,w,b,delta = 1.0):
    m = len(y)
    y_predict = x_poly.dot(w) + b
    erros = y_predict - y

    #Huber loss
    mask = np.abs(erros) <= delta
    loss = np.where(mask, (1/2)*erros**2, delta*np.abs(erros) - (1/2)*delta**2)
    cost = np.sum(loss)/m

    #rearranging erro measure
    gradient_erro = np.where(mask,erros, delta*np.sign(erros))
    
    dj_dw = np.dot(x_poly.T,gradient_erro) * (1/m)      #multiplying matrix transposed x by their respective gradient
    dj_db = np.sum(gradient_erro) * (1/m)

    return cost,dj_dw,dj_db
