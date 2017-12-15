from os import listdir
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def midpoints(inlist):
    out = []
    inlist = np.sort(inlist)
    if len(inlist) > 0:
        for i in range(len(inlist)-1):
            out.append(np.mean(inlist[i:i+2]))
    return out

def distance(x, labels, feature_set):
    x = x[:,feature_set]
    sd = np.std(x, ddof=1, axis=0, keepdims=True)
    x = x / sd
    centre = np.mean(x[labels,:], axis=0, keepdims=True)
    return sd, centre, np.sqrt(np.sum((x - centre) ** 2, axis=1))

def roc_auc(data_train, feature_set):
    x = data_train[:,:-1]
    labels = np.array(data_train[:,-1], dtype=bool)
    _, _, dist = distance(x, labels, feature_set)
    low, high = np.min(dist), np.max(dist)
    pred = (high - dist) / (high - low)
    return roc_auc_score(y_true=labels, y_score=pred)

def select(data_train):
    selected = []
    remaining = list(range(data_train.shape[1]-1))
    old_score = -1
    new_score = 0
    while True:
        roc_scores = np.zeros((len(remaining), 2))
        for i,j in enumerate(remaining):
            feature_set = selected + [j]
            roc_scores[i,:] = [roc_auc(data_train, feature_set), j]
        new_score = np.max(roc_scores[:,0])
        pos = int(roc_scores[np.argmax(roc_scores[:,0]),1])
        if new_score > old_score:
            old_score = new_score
            selected += [pos]
            remaining.remove(pos)
        else:
            break
    return selected, old_score

def threshold(data_train, feature_set):
    x = data_train[:,:-1]
    labels = np.array(data_train[:,-1], dtype=bool)
    sd, centre, dist = distance(x, labels, feature_set)
    thresholds = midpoints(dist)
    accuracies = np.zeros((len(thresholds), 2))
    for i,t in enumerate(thresholds):
        pred = np.less(dist, t)
        accuracy = accuracy_score(labels, pred)
        accuracies[i,:] = [accuracy, t]
    best = np.max(accuracies[:,0])
    pos = len(thresholds) - np.argmax(accuracies[:,0][::-1]) - 1 # choose the biggest one
    best_t = accuracies[pos,1]
    return best, best_t, centre, sd

if __name__ == '__main__':
    tightframe = True
    #filelist = listdir('pics')

    disputed_id = np.array([1, 7, 10, 20, 23, 25, 26]) - 1
    r = np.asarray([2, 3, 4, 5, 6, 8, 9, 21, 22, 24, 27, 28]) - 1
    nr = np.array([11, 12 ,13, 14, 15, 16, 17, 18, 19]) - 1#, 29, 30 ,31, 32, 33, 34, 35]) - 1
    confirmed = np.sort(np.append(r, nr))

    if tightframe:
        features = np.loadtxt('features.txt')[:28,:]
        n_files, n_features = features.shape
        x_raw = np.zeros((n_files, n_features+1))
        x_raw[:,:-1] = features
        x_raw[r,-1] = 1
        data = x_raw[confirmed,:]
        n_samples = data.shape[0]
    else:
        features = np.transpose(np.loadtxt('color_feature .txt'))
        n_samples, n_features = features.shape
        data = np.zeros((n_samples, n_features+1))
        data[:,:-1] = features
        data[:,-1] = [1] * 7 + [0] * 9 + [1] * 5 + [0] * 7

    # Cross validation
    results = []
    positive = []
    negative = []
    feature_dict = dict()
    use_most_frequent = True
    for i in range(n_samples):
        print('Cross validation iteration: ', i+1)
        data_train = data[[j for j in range(n_samples) if j != i],:]
        data_test = data[i,:-1]
        label_test = np.array(data[i,-1], dtype=bool)

        if not use_most_frequent:
            feature_set, score = select(data_train)
            for f in feature_set:
                if f in feature_dict:
                    feature_dict[f] += 1
                else:
                    feature_dict[f] = 1
            print('Selected feature(s): ', feature_set, \
                'Corresponding ROC_AUC score: ', score)
        else:
            feature_set = np.sort(np.array([6, 36, 49, 0]))

        a, t, centre, sd = threshold(data_train, feature_set)
        print('Training accuracy: ', a, 'Threshold: ', t)
        data_test = data_test[feature_set]
        data_test = data_test / sd.flatten()
        pred_score = np.sqrt(np.sum((data_test - centre.flatten()) ** 2))
        pred = np.less(pred_score, t)
        result = int(pred == label_test)
        print('Prediction score: ', pred_score, 'Test result: ', result, '\n')
        results.append(result)
        if label_test:
            positive.append(result)
        else:
            negative.append(result)

    print('Cross validation accuracy: ', np.mean(results))
    if not use_most_frequent:
        print('Feature frequency: ', feature_dict)
    print('Positives: ', positive)
    print('TPR: ', np.mean(positive))
    print('Negatives: ', negative)
    print('TNR: ', np.mean(negative))

    # Make predictions on the disputed ones
    data_test = x_raw[disputed_id,:-1]
    data_train = x_raw[confirmed,:]
    feature_set, score = select(data_train)
    print('Selected feature(s): ', feature_set, 'Corresponding ROC_AUC score: ', score)
    a, t, centre, sd = threshold(data_train, feature_set)
    print('Training accuracy: ', a, 'Threshold: ', t)
    data_test = data_test[:,feature_set]
    data_test = data_test / sd
    pred_score = np.sqrt(np.sum((data_test - centre) ** 2, axis=1))
    pred = np.less(pred_score, t)
    print('Prediction scores: ', pred_score, 'Predictions: ', pred)
