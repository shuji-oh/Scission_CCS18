from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedKFold

def main():
    df = pd.read_csv("voltage.csv")
    X = df[['mean_G00','stdev_G00','variance_G00','skew_G00','kurtosis_G00','max_G00','min_G00','rms_G00','en_G00','mean_fft_G00','stdev_fft_G00','variance_fft_G00','skew_fft_G00','kurtosis_fft_G00','max_fft_G00','min_fft_G00','rms_fft_G00','en_fft_G00','mean_G01','stdev_G01','variance_G01','skew_G01','kurtosis_G01','max_G01','min_G01','rms_G01','en_G01','mean_fft_G01','stdev_fft_G01','variance_fft_G01','skew_fft_G01','kurtosis_fft_G01','max_fft_G01','min_fft_G01','rms_fft_G01','en_fft_G01','mean_G10','stdev_G10','variance_G10','skew_G10','kurtosis_G10','max_G10','min_G10','rms_G10','en_G10','mean_fft_G10','stdev_fft_G10','variance_fft_G10','skew_fft_G10','kurtosis_fft_G10','max_fft_G10','min_fft_G10','rms_fft_G10','en_fft_G10']]
    Y = df['label'].map({'ECU1': 1, 'ECU2': 2, 'ECU3': 3, 'ECU4': 4, 'ECU5': 5, 'ECU6':6 })
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # dividing the data to 80% as training data, 20% as testing data
    
    # rf
    rf = RandomForestClassifier(n_estimators=50)

    # K-fold cross validation
    #print(np.mean(cross_val_score(rf, X, Y, cv=5)))
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cvscores = []
    X = X.values
    Y = Y.values
    for train, test in kfold.split(X, Y):
        rf.fit(X[train], Y[train])
        Y_pred = rf.predict(X[test])
        scores = accuracy_score(Y[test], Y_pred)
        cvscores.append(scores*100)
        print('Test accuracy:', scores)

    # K-fold result
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    C = confusion_matrix(Y_test, Y_pred)

    # Normalization
    NC = C / C.astype(np.float).sum(axis=1)
    print(NC)

    # plot
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((Y_train, Y_test))
    df = pd.DataFrame(NC, index=['ECU0', 'ECU1', 'ECU2', 'ECU3', 'ECU4', 'ECU5', 'ECU6'], columns=['ECU0', 'ECU1', 'ECU2', 'ECU3', 'ECU4', 'ECU5', 'ECU6'])
    fig = plt.figure()
    sns.heatmap(df, cmap="Greens", annot=True, fmt=".4f")
    plt.yticks(rotation=0)
    
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')

    pp = PdfPages('rf_confusion.pdf')
    pp.savefig(fig)

    pp.close()
    plt.close('all')

if __name__ == '__main__':
    main()
