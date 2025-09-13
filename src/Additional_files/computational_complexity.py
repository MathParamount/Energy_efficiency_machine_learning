from src.main import *
from polinomial_function import *
from data_visualization import *

import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def runtimes_details(x_train_poly_clf):

    #Regression model
    start = time.time()
    model = LinearRegression()

    predict = model.predict(x_test_reg)
    end = time.time()

    latency = end - start

    print(f"The latency of regression model is: {latency} seconds") 

    #Classification model
    start_clf = time.time()
    model = LogisticRegression(x_train_poly_clf)

    predict = model.predict(x_test_clf)
    end_clf = time.time()

    latency_clf = end_clf - start_clf

    print(f"The latency of classification model is: {latency_clf} seconds")
