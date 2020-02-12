from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import struct
import itertools
import numpy as np
from sklearn import neighbors, metrics
import matplotlib.pyplot as plt
import time
start_time = time.time()

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


#This code converts the MNIST idx format into numpy arrays
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

#import the train and test data and labels. Using all 60,000 images for training and 10,000 images for testing
raw_train = read_idx("train-images-idx3-ubyte")
train_data = np.reshape(raw_train, (60000, 28*28))
train_label = read_idx("train-labels-idx1-ubyte")

raw_test = read_idx("t10k-images-idx3-ubyte")
test_data = np.reshape(raw_test, (10000, 28*28))
test_label = read_idx("t10k-labels-idx1-ubyte")
#selecting all the data, digits 0-9 for training
idx = (train_label ==0) | (train_label == 1) | (train_label == 2) | (train_label == 3) | (train_label == 4) | (train_label == 5) | (train_label == 6) | (train_label == 7) | (train_label == 8) | (train_label == 9)
x = train_data[idx]
y = train_label[idx]


#creating the RANDOM FOREST classifier and setting paramters, either gini or entropy criterion. As well as depth. Default depth, runs until leaf is pure
# x, y = make_classification(n_features=784)
# DT = RandomForestClassifier(criterion='gini', n_jobs=-1).fit(x, y)
DT = RandomForestClassifier(criterion='entropy', max_depth = None, n_jobs=-1)
RndTree = DT.fit(x,y)
print (RndTree)

#Setting the test Label
idx =  (test_label == 0) | (test_label == 1) | (test_label == 2) | (test_label == 3) | (test_label == 4) | (test_label == 5) | (test_label == 6) | (test_label == 7) | (test_label == 8) | (test_label == 9)
x_test = test_data[idx]
y_true = test_label[idx]
y_pred = DT.predict(x_test)

import itertools

#Create confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix Random forest with entropy criterion max depth = none',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#using thee sklearn score function, we find the mean accuracy of the decision tree
score = RandomForestClassifier.score(DT, x_test, y_true)
scorestr = str(score)
print("Random Forest mean accuracy score using entropy criterion and max depth = none is: " + scorestr)


#plots confusion matrix non-normalized
cm = metrics.confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
plt.show()

#plots confusion matrix normalized
plot_confusion_matrix(cm, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], normalize=True)
plt.show()



print("--- %s seconds ---" % (time.time() - start_time))