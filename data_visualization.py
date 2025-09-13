import matplotlib.pyplot as plt

#regression libraries
from pre_training import *
from src.cost_gradient_huber import *
from adam_compute import *
from src.polinomial_function import *

#classification libraries
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer      #Transform the continuos data in discrete
from sklearn.model_selection import GridSearchCV

#Additional file
#from src.Additional_files.computational_complexity import *

#Regression model
try:
    num_iter = int(input("write down the iteration numbers: "))
    alpha = float(input("Type the learning rate: "))

    #Verifying shapes
    print(f"\ny_test_reg shape: {y_test_reg.shape}")
    print(f"y_train_reg shape: {y_train_reg.shape}\n")

    # Apply polynomial transformation
    x_train_poly = polinomial_compute(x_train_reg)
    x_test_poly = polinomial_compute(x_test_reg)

    y_train_reg = np.ravel(y_train_reg)  # Transform (614, 1) → (614,)
    y_test_reg = np.ravel(y_test_reg) 

    #Verification of dimension
    print(f"after redimension - y_train_reg shape: {y_train_reg.shape}\n")
    print(f"after redimension - y_test_reg shape: {y_test_reg.shape}\n")

    #initializations to all features
    b = 0
    w = np.zeros(x_train_poly.shape[1])
    
    # Train weights
    w_adam,b_adam,cost_history = adam_correlation(x_train_poly,y_train_reg,w,b,alpha,num_iter,delta = 1.5)

    #Veryfing if the sample number is consistent
    if x_test_poly.shape[0] != len(y_test_reg):
        min_test_samples = min(x_test_poly.shape[0], len(y_test_reg))
        x_test_poly = x_test_poly[:min_test_samples]
        y_test_reg = y_test_reg[:min_test_samples]

        print(f"\nWe are using the first sample2: {min_test_samples} to match\n")

    # Prediction
    y_pred_reg = x_test_poly.dot(w_adam) + b_adam


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_test_reg)), y_test_reg, label='True', alpha=0.7, s=20)
    plt.scatter(range(len(y_pred_reg)), y_pred_reg, label='Predicted', alpha=0.7, s=20)
    plt.title('Polynomial Regression: True vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Cooling Load')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Correlation')
    
    plt.tight_layout()
    plt.savefig("regression_plot.png")
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()

#Classification model
try:
    x_train_poly_clf = polinomial_compute(x_train_clf)
    x_test_poly_clf = polinomial_compute(x_test_clf)
    
    #Treating the data to ordinary type
    est = KBinsDiscretizer(n_bins=5, encode = 'ordinal', strategy= 'uniform')
    y_train_clf_discrete = est.fit_transform(y_train_clf.reshape(-1,1)).astype(int).ravel()
	
	#Searching the better C value to penalty
    parm_c = {'C' : [0.01,0.1,1,10], 'solver': ['lbfgs']}
    grid = GridSearchCV(LogisticRegression(),parm_c, cv =5)
    grid.fit(x_train_poly_clf,y_train_clf_discrete)
    print("The best c parameter is", grid.best_params_['C'])
    
    #scaling the data

    scaler = StandardScaler()
    x_train_pclf_scaler = scaler.fit_transform(x_train_poly_clf)
    x_test_pclf_scaler = scaler.fit_transform(x_test_poly_clf)

    #Training model
    clf_model = LogisticRegression(class_weight= 'balanced', penalty = 'l2', C = grid.best_params_['C'], multi_class='multinomial',solver = 'saga',max_iter= 5000,tol=1e-4)
    clf_model.fit(x_train_poly_clf,y_train_clf_discrete)
    
    #prediction
    y_pred_clf = clf_model.predict(x_test_poly_clf)
    y_pred_prob = clf_model.predict_proba(x_test_poly_clf)

    #Treating the y_test and y_pred
    y_test_clf_discrete = est.transform(y_test_clf.reshape(-1,1)).astype(int).ravel()
    y_pred_clf_discrete = est.transform(y_pred_clf.reshape(-1,1)).astype(int).ravel()

    #Exactness of the model
    accuracy = 100*(np.sum(y_test_clf_discrete == y_pred_clf_discrete))/len(y_test_clf_discrete)
    print(f"\n\nThe accuracy of this classification model: {round(accuracy,3)}%\n")

    #confusion matrix
    conf_matrix = confusion_matrix(y_test_clf_discrete,y_pred_clf_discrete, normalize = 'true')
    print("Confusion matrix of prediction model")
    print(conf_matrix)
    
    #naming each class
    classes = ['low', 'medium','high']

    # Plotting the data
    plt.figure(figsize=(10,8))

    # Plot true values
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_test_clf)), y_test_clf, alpha=0.7, label='True', c='blue')
    plt.scatter(range(len(y_pred_clf)), y_pred_clf, alpha=0.7, label='Predicted', c='red', marker='x')
    plt.xlabel('Sample')
    plt.ylabel('Energy Efficiency Class')
    plt.yticks([0, 1, 2], classes)
    plt.title('True vs Predicted Values')
    plt.legend()

    # Plot errors
    plt.subplot(1, 2, 2)
    errors = y_pred_clf - y_test_clf

    print(f'\nerror shape: {errors}\n')

    plt.hist(errors.flatten(), bins=7, alpha=0.7, color='yellow')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()


y_pred_reg = x_test_poly.dot(w_adam) + b_adam

#Residuo compute
residuo = y_pred_reg - y_test_reg

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.scatter(y_pred_reg,residuo, alpha=0.5)
plt.axhline(y=0,color='r', linestyle='--')
plt.xlabel('prediction')
plt.ylabel('residuo')
plt.title('Residuo vs prediciton')

plt.subplot(1,2,2)
plt.scatter(y_test_reg,y_pred_reg,alpha=0.5)
plt.plot([y_test_reg.min(),y_test_reg.max()] , [y_test_reg.min(),y_test_reg.max()])
plt.xlabel('real values')
plt.ylabel('predictions')
plt.title('True vs prediction')

plt.tight_layout()
plt.show()



print('..........................')

#optional = input("Do you want to see latency time? (s or n)")

#if(optional == 's'):
    #runtimes_details(x_train_poly_clf)
    
