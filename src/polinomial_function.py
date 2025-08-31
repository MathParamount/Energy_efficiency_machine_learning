from pre_training import *

def polinomial_compute(x_train_norm):
   x1 = x_train_norm
   x2 = x_train_norm**2
   x3 = x_train_norm**3
   x4 = x_train_norm**4
   x5 = x_train_norm**5
   x6 = x_train_norm**6

   x_poly = np.column_stack([x1,x2,x3,x4,x5,x6])

   return x_poly
