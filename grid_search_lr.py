import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

def main():
    df = pd.read_csv("voltage.csv")
    
    # Full features
    X = df[['mean_G00','stdev_G00','variance_G00','skew_G00','kurtosis_G00','max_G00','min_G00','rms_G00','en_G00','mean_fft_G00','stdev_fft_G00','variance_fft_G00','skew_fft_G00','kurtosis_fft_G00','max_fft_G00','min_fft_G00','rms_fft_G00','en_fft_G00','mean_G01','stdev_G01','variance_G01','skew_G01','kurtosis_G01','max_G01','min_G01','rms_G01','en_G01','mean_fft_G01','stdev_fft_G01','variance_fft_G01','skew_fft_G01','kurtosis_fft_G01','max_fft_G01','min_fft_G01','rms_fft_G01','en_fft_G01','mean_G10','stdev_G10','variance_G10','skew_G10','kurtosis_G10','max_G10','min_G10','rms_G10','en_G10','mean_fft_G10','stdev_fft_G10','variance_fft_G10','skew_fft_G10','kurtosis_fft_G10','max_fft_G10','min_fft_G10','rms_fft_G10','en_fft_G10']]

    # Selected features by Relief-F
    #X = df[['stdev_G01','kurtosis_G01','variance_G01','en_G00','rms_G00','mean_G00','mean_G10','en_G10','rms_G10','min_G01','mean_G01','min_G10','rms_G01','en_G01','max_G00','max_G10','mean_fft_G10','mean_fft_G01']]

    Y = df['label'].map({'ECU1': 1, 'ECU2': 2, 'ECU3': 3, 'ECU4': 4, 'ECU5': 5, 'ECU6':6 })
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # dividing the data to 80% as training data, 20% as testing data
    
    # Grid Search
    param_grid = {"penalty": ['l2'],
                  "C":[0.001, 0.01, 0.1, 1.0, 10],
                  "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  "max_iter": [100, 1000, 10000],
                  "tol": [0.0001, 0.001]}

    forest_grid = GridSearchCV(estimator = LogisticRegression(random_state=0),
                               param_grid = param_grid,   
                               scoring = "accuracy", #metrics
                               cv = 5,               #cross-validation
                               n_jobs = 4)           #number of core

    forest_grid.fit(X,Y) #fit

    forest_grid_best = forest_grid.best_estimator_ #best estimator
    print("Best Model Parameter: ", forest_grid.best_params_)

    GS_penalty = forest_grid.best_params_['penalty']
    GS_C = forest_grid.best_params_['C']
    GS_solver = forest_grid.best_params_['solver']
    GS_max_iter = forest_grid.best_params_['max_iter']
    GS_tol = forest_grid.best_params_['tol']

    # lr
    lr = LogisticRegression(penalty=GS_penalty, C=GS_C, solver=GS_solver, max_iter=GS_max_iter, tol=GS_tol)

    # K-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cvscores = []
    X = X.values
    Y = Y.values
    for train, test in kfold.split(X, Y):
        lr.fit(X[train], Y[train])
        Y_pred = lr.predict(X[test])
        scores = accuracy_score(Y[test], Y_pred)
        cvscores.append(scores*100)
        print('Test accuracy:', scores)
    # K-fold result
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    # plot heatmap 
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    C = confusion_matrix(Y_test, Y_pred)
    # Normalization
    NC = C / C.astype(np.float).sum(axis=1)
    print(NC)
    # plot
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((Y_train, Y_test))
    
    df = pd.DataFrame(NC, index=['ECU1', 'ECU2', 'ECU3', 'ECU4', 'ECU5', 'ECU6'], columns=['ECU1', 'ECU2', 'ECU3', 'ECU4', 'ECU5', 'ECU6'])
    fig = plt.figure()
    sns.heatmap(df, cmap="Greens", annot=True, fmt=".4f")
    plt.yticks(rotation=0)
    
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')

    pp = PdfPages('lr_confusion.pdf')
    pp.savefig(fig)

    pp.close()
    plt.close('all')

if __name__ == '__main__':
    main()
