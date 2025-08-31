from src.main import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#pre-training normalize
y_train = df['carga de resfriamento']

x_train = df.drop('carga de resfriamento', axis=1)

#normalizing fuction
def compute_normalizing_function(X):
    standard = StandardScaler()
    transf = standard.fit_transform(X)

    return transf

x_train_norm = compute_normalizing_function(x_train.values)
y_train_norm = compute_normalizing_function(y_train.values.reshape(-1,1))

#Classifing the target
for col in cols:
    df[f"{col}_class"] = pd.cut(df[col], bins=3, labels=['low','medium','high'])

#splitting the data
x_train_reg, x_test_reg,y_train_reg,y_test_reg = train_test_split(x_train_norm,y_train_norm,test_size= 0.2, random_state=42)

x_test_clf,x_train_clf,y_train_clf,y_test_clf = train_test_split(x_train_norm,y_train_norm,test_size= 0.2,random_state= 42)

#Transposed adjust
x_train_clf = x_train_clf.T
x_test_clf = x_test_clf.T
