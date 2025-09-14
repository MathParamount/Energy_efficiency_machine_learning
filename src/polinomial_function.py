from pre_training import *
from sklearn.preprocessing import PolynomialFeatures

def polinomial_compute(x_poly, max_degree = 3, num_total_features= 1000):   
   
   poly = PolynomialFeatures(degree= max_degree, include_bias= False)

   x_poly_feature = poly.fit_transform(x_poly,)

   if x_poly_feature.shape[1] > num_total_features:
        x_poly_feature = x_poly_feature[:, :num_total_features]

   return x_poly_feature