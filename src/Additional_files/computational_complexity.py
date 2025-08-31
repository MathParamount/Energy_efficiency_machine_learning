from src.main import *
from polinomial_function import *
from data_visualization import *

import time
from sklearn.linear_model import LinearRegression

def runtimes_details():

    #Regression model
    start = time.time()
    model = LinearRegression()

    predict = model.predict(x_test_reg)
    end = time.time()

    latency = end - start

    print(f"The latency of regression model is: {latency} seconds") 

    #Classification model
    start_clf = time.time()
    model = LogisticRegression()

    predict = model.predict(x_test_clf)
    end_clf = time.time()

    latency_clf = end_clf - start_clf

    print(f"The latency of classification model is: {latency_clf} seconds")