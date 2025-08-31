from src.cost_gradient_huber import *

def adam_correlation(x_poly,y,w,b,alpha,num_iter,delta = 1.0):

    #Geral function to prediction of target
    y_pred = x_poly.dot(w) + b
    
    #Moment and smrProp of each gradient
    m_dw = np.zeros_like(w)
    m_db = 0
    s_dw = np.zeros_like(w)
    s_db = 0

    #default variables
    epsilon = 1e-8
    b1 = 0.9
    b2 = 0.999

    step = 0

    cost_history = []

    for i in range(num_iter):
        
        step += 1

        cost,dw,db = gradient_huber(x_poly,y,w,b,delta)        

        m_dw = b1*m_dw + (1 - b1)*dw
        m_db = b1*m_db + (1 - b1)*db

        s_dw = b2*s_dw + (1- b2)*(dw**2)
        s_db = b2*s_db + (1 -b2)*(db**2)

        #moment bias correction
        m_dw_correction = m_dw / ( 1 - b1**(num_iter+1))
        m_db_correction = m_db / ( 1 - b1**(num_iter+1))

        #smrProp bias correction
        s_dw_correction = s_dw / ( 1 - b2**(num_iter+1))
        s_db_correction = s_db / ( 1 - b2**(num_iter+1))

        w -= alpha * m_dw_correction / (np.sqrt(s_dw_correction) + epsilon)
        b -= alpha * m_db_correction / (np.sqrt(s_db_correction) + epsilon)

        cost_history.append(cost)               #Getting historic of cost

        if step % 100 == 0:
            print(f"Iteration {step}, Cost: {cost:.6f}")

    return w,b,cost_history
