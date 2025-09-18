from pre_training import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

def polinomial_compute_fit_reg(x_poly, max_degree = 3, num_total_features= 1000):   
   
   poly = PolynomialFeatures(degree= max_degree, include_bias= False)

   x_poly_feature = poly.fit_transform(x_poly)

   if x_poly_feature.shape[1] > num_total_features:
        selector = SelectKBest(score_func= f_regression, k = num_total_features)
        x_poly_feature = selector.fit_transform(x_poly_feature, y_train_reg)

        return x_poly_feature,poly, selector
   else:
       return x_poly_feature,poly,None

def polinomial_compute_transform_reg(x_poly, poly, selector = None):
    X_poly = poly.transform(x_poly)

    if selector:
        X_poly = selector.transform(X_poly)

    return X_poly

#Classification

def polinomial_compute_fit_clf(x_poly, y = None, max_degree = 3, num_total_features= 1000):
    

def polinomial_compute_transform_clf(x_poly, poly, selector = None):
