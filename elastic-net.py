import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import ElasticNet
from sklearn.metrics import mean_squared_error

def modelElastic(alpha=1):
        modelElastic= ElasticNet(random_state=0, alpha=alpha)
        modelElastic.fit(X_train, y_train)
        y_predic_elastic=modelElastic.predict(X_test)
        # loss function
        elastic_loss = mean_squared_error(y_test, y_predic_elastic)
        return elastic_loss

if __name__ == '__main__':
    alphas = np.arange(0,1,0.01)
    loss_total = []
    for i in alphas:
        res = modelElastic(i)
        loss_total.append(res)

    loss_total = np.array(loss_total)
    plt.plot(alphas, loss_total)
    plt.xlabel('alphas')
    plt.ylabel('Loss Elastic')
    plt.text(0.02, 0.8, 'loss min:{}'.format(np.min(loss_total)), fontsize=7)
    plt.show()