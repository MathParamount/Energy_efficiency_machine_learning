from pre_training import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

#Regression

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
    
    poly = PolynomialFeatures(degree= max_degree)
    x_poly = poly.fit_transform(x_poly)

    if x_poly.shape[1] > num_total_features and y is not None:
        
        selector = SelectKBest(score_func= f_classif, k = num_total_features)
        x_poly = selector.fit_transform(x_poly, y)

        return x_poly , {'poly', poly, 'selector', selector}
    
    else:

        return x_poly, {'poly': poly, 'selector': None}
    
def polinomial_compute_transform_clf(x_poly, transformer_dict):

    x_poly = transformer_dict['poly'].transform(x_poly)

    if(transformer_dict['selector']):
        x_poly = transformer_dict['selector'].transform(x_poly)

    return x_poly