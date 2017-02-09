#!/usr/bin/python
# -*- coding: UTF-8 -*-



def getDataFromURl():
    import urllib2
    url = 'http://aima.cs.berkeley.edu/data/iris.csv'
    u = urllib2.urlopen(url)
    localFile = open('iris.csv', 'w')
    localFile.write(u.read())
    localFile.close()

def createDataModel():
    from numpy import genfromtxt
    # read the first 4 columns
    data = genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    # read the fifth column
    target = genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
    target2 = genfromtxt('iris.csv', delimiter=',', usecols=(2,3))
    print data.shape
    print target.shape
    print set(target)  # build a collection of unique elements
    print target2 <= 1.4

def createPlot():
    from pylab import plot, show
    from numpy import genfromtxt
    data = genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    target = genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)

    plot(data[target == 'setosa', 0], data[target == 'setosa', 2], 'bo')
    plot(data[target == 'versicolor', 0], data[target == 'versicolor', 2], 'ro')
    plot(data[target == 'virginica', 0], data[target == 'virginica', 2], 'go')
    show()

def createFigure():
    from numpy import genfromtxt
    from pylab import figure, subplot, hist, xlim, show
    data = genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    target = genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
    xmin = min(data[:, 0])
    xmax = max(data[:, 0])
    figure()
    subplot(411)  # distribution of the setosa class (1st, on the top)
    hist(data[target == 'setosa', 0], color='b', alpha=.7)
    xlim(xmin, xmax)
    subplot(412)  # distribution of the versicolor class (2nd)
    hist(data[target == 'versicolor', 0], color='r', alpha=.7)
    xlim(xmin, xmax)
    subplot(413)  # distribution of the virginica class (3rd)
    hist(data[target == 'virginica', 0], color='g', alpha=.7)
    xlim(xmin, xmax)
    subplot(414)  # global histogram (4th, on the bottom)
    hist(data[:, 0], color='y', alpha=.7)
    xlim(xmin, xmax)
    show()

def createClassifier():
    from numpy import genfromtxt, zeros, mean
    data = genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    target = genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)

    t = zeros(len(target))
    t[target == 'setosa'] = 1
    t[target == 'versicolor'] = 2
    t[target == 'virginica'] = 3

    from sklearn import cross_validation
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    classifier = GaussianNB()
    train, test, t_train, t_test = cross_validation.train_test_split(data, t,test_size = 0.4, random_state = 0)
    classifier.fit(train, t_train)  # train
    precision = classifier.score(test, t_test)  # test
    cmatrix = confusion_matrix(classifier.predict(test), t_test)
    classifyReport = classification_report(classifier.predict(test), t_test, target_names=['setosa', 'versicolor', 'virginica'])
    # cross validation with 6 iterations
    scores = cross_validation.cross_val_score(classifier, data, t, cv=6)

    print precision
    print cmatrix
    print classifyReport
    print scores
    print mean(scores)


if __name__ == '__main__':
    createClassifier()
