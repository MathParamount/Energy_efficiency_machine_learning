from src.main import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy import stats         #Para box-cox adjust

#pre-training normalize
col = 'carga de resfriamento'

y_train = df[col]
x_train = df.drop(col, axis=1)

#Original and box-cox of data
plt.subplot(1,2,1)

plt.title('Original data')
sea.histplot(y_train)
plt.xlabel('value')
plt.ylabel('frequency')

#Box-cox
y_transf_data, lambda_value = stats.boxcox(y_train)
print(f'optimal lambda value: {lambda_value}')

plt.subplot(1,2,2)
plt.title('Box-cox data')
sea.histplot(y_transf_data,kde = 'True')
plt.xlabel('value')
plt.ylabel('frequency')
plt.show()

#normalizing fuction
def compute_normalizing_function(X):
    standard = StandardScaler()
    transf = standard.fit_transform(X)

    return transf

x_train_norm = compute_normalizing_function(x_train.values)
y_train_norm = compute_normalizing_function(y_transf_data.reshape(-1,1))


#Verifying class balance
y_train_discrete = pd.cut(df[col], bins = 3, labels = [0,1,2]).astype(int)
print(np.bincount(y_train_discrete))

#Classifing the target

df['carga de resfriamento'] = pd.cut(y_train, bins=3, labels=['low','medium','high'])

#splitting the data
x_train_reg, x_test_reg,y_train_reg,y_test_reg = train_test_split(x_train_norm,y_train_norm,test_size= 0.25, random_state=42)

x_train_clf,x_test_clf,y_train_clf,y_test_clf = train_test_split(x_train_norm,y_train_norm,test_size= 0.25,random_state= 42)

#Transposed adjust
#x_train_clf = x_train_clf.T
#x_test_clf = x_test_clf.T
