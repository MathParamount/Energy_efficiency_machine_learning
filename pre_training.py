from src.main import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy import stats         #To box-cox adjust

#pre-training normalize
col = 'carga de resfriamento'

target = df[col]
data_ = df.drop(col, axis=1)

#Train-test split
x_train, x_test,y_train,y_test = train_test_split(data_,target,test_size= 0.25, random_state=42)

#Box-cox
y_transf_data, lambda_value = stats.boxcox(y_train + 1e-10)
print(f'optimal lambda value: {lambda_value}')

#Applying validation
y_test_transf = stats.boxcox(y_test + 1e-10, lmbda = lambda_value)

#Original and box-cox of data visualization

plt.subplot(1,2,1)
plt.title('Original data')
sea.histplot(y_train)
plt.xlabel('value')
plt.ylabel('frequency')

plt.subplot(1,2,2)
plt.title('Box-cox data')
sea.histplot(y_transf_data, kde = 'True')
plt.xlabel('value')
plt.ylabel('frequency')
plt.show()


#normalizing fuction
def compute_normalizing_function_fit(X):
    standard = StandardScaler()
    transf = standard.fit_transform(X)

    return transf,standard

def compute_normalizing_function_transform(x,scaler):
    return scaler.transform(x)


#X train and your validation
x_train_norm, scaler_x = compute_normalizing_function_fit(x_train.values)
x_test_norm = compute_normalizing_function_transform(x_test,scaler_x)

#y train your validation
y_train_norm, scaler_y = compute_normalizing_function_fit(y_test_transf.reshape(-1,1))
y_test_norm = compute_normalizing_function_transform(y_test_transf.reshape(-1,1), scaler_y)

#Building regression variables
x_train_reg, y_train_reg = x_train_norm, y_train_norm.flatten()
x_test_reg, y_test_reg = x_test_norm, y_test_norm.flatten()


#Building classificaition variables
y_train_clf = np.digitize(y_train, bins = [np.percentile(y_train,33), np.percentile(y_train,66)])
y_test_clf = np.digitize(y_test, bins = [np.percentile(y_train,33), np.percentile(y_train,66)])

x_train_clf, y_train_clf = x_train_norm, y_train_norm
x_test_clf, y_test_clf = x_test_norm, y_test_norm

#Verifying class balance
y_train_discrete = pd.cut(df[col], bins = 3, labels = [0,1,2]).astype(int)
print(np.bincount(y_train_discrete))

#Verifying if there is not a leak
print(f'The mean of x_train: {np.mean(x_train_norm, axis = 0)[:3]}')
print(f'The mean of x_test: {np.mean(x_test_norm, axis = 0)[:3]}')

#Number of features

print(f'Number of features: {x_train_reg.shape[1]}')
#Classifing the target

df['carga de resfriamento'] = pd.cut(y_train, bins=3, labels=['low','medium','high'])

