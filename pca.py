import pandas as pd 
import sklearn
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    data_heart = pd.read_csv('./datasets/heart_bde64b4c-2d72-4cd3-a964-62ee94855f5b.csv')
    data_features = data_heart.drop(['target'],axis=1)
    data_target = data_heart['target']
    data_features = StandardScaler().fit_transform(data_features)
    x_train,x_test,y_train,y_test = train_test_split(data_features,data_target,test_size=.3,random_state=42)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    pca = PCA(n_components=4)
    pca.fit(x_train)

    ipca = IncrementalPCA(n_components=4,batch_size=10)
    ipca.fit(x_train)
    

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    print(pca.explained_variance_,pca.explained_variance_ratio_)


    logistic = LogisticRegression(solver='lbfgs')
    dt_train = pca.transform(x_train)
    dt_test = pca. transform(x_test)
    logistic.fit(dt_train, y_train)
    print('SCORE PCA:',logistic.score(dt_test,y_test))

    dt_train = ipca.transform(x_train)
    dt_test = ipca. transform(x_test)
    logistic.fit(dt_train, y_train)
    print('SCORE IPCA:',logistic.score(dt_test,y_test))

    kpca = KernelPCA(n_components=4,kernel='poly')
    kpca.fit(x_train)
    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train,y_train)
    print('SCORE KPCA:',logistic.score(dt_test,y_test))