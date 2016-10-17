from numpy import *
import random as rnd
import matplotlib.pyplot as plt
from collections import Counter
MAX_ITERATIONS = 4


def getRandomCentroids(dataset, k):
    numFeatures = 1
    numData = shape(dataset)[0]
    if ndim(dataset) > 1:
        numFeatures = shape(dataset)[1]

    # Assign k sample as centroids
    return array(rnd.sample(dataset, k))


def getCentroids(dataset, labels, k):
    newCentroids = []
    for xcluster in range(k):
        cluster = [i for i, j in zip(dataset, labels) if j == xcluster]
        newCentroids.append(mean(cluster, axis=0))
    # print "{0:-^30}".format("New Centrolds")
    # print newCentroids
    return array(newCentroids)


def distance(d1, d2):
    assert len(d1) == len(d2)
    d = len(d1)
    return sqrt(sum([(i - j) * (i - j) for i, j in zip(d1, d2)]))


def getLabels(dataset, centroids):
    k = len(centroids)
    labels = zeros((len(dataset)))
    for dindex, datapoint in enumerate(dataset):
        min_distance = float('inf')
        label = -1
        for cindex, c in enumerate(centroids):
            d = distance(datapoint, c)
            label = cindex if min_distance > d else label
            min_distance = d if min_distance > d else min_distance

        labels[dindex] = label
    return labels


def shouldStop(oldCentroids, centroids, iterations):
    if iterations >= MAX_ITERATIONS:
        return True
    if oldCentroids is None:
        return False
    return array_equal(oldCentroids, centroids)


def kmeans(dataset, k):
    # Initialize centroids randomly ( Forgy approch)
    numFeatures = 1
    if ndim(dataset) > 1:
        numFeatures = shape(dataset)[1]
    centroids = getRandomCentroids(dataset, k)
    # print centroids
    # figure = plt.figure()
    # ax1 = figure.add_subplot(111)

    # plt.plot(dataset[:, 0], dataset[:, 1], 'bo')
    # plt.plot(centroids[:, 0], centroids[:, 1], 'w.')

    # plt.show()

    # Initialize book keeping vars
    iterations, oldCentroids = 0, None
    labels = None
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids
        oldCentroids = centroids
        iterations += 1

        # Assign labels to each data point based on centroids
        labels = getLabels(dataset, centroids)

        # Assign new centroids base on datapoint label
        # figure2 = plt.figure()
        # ax2 = figure2.add_subplot(111)
        '''
        color = ['r', 'b', 'g', 'c', 'm', 'i', 'k', 'w']
        for i in range(k):  # For each
            # print labels
            cluster = []
            for j in range(len(dataset)):
                if labels[j] == i:
                    cluster.append(dataset[j])
            if len(cluster) > 0:
                cluster = array(cluster)
                plt.plot(cluster[:, 0], cluster[:, 1], color[i] + 'o')
        plt.plot(centroids[:, 0], centroids[:, 1], 'k^')'''
        # Calculate new centroids
        centroids = getCentroids(dataset, labels, k)
    # plt.show()
    return labels

def calcInformationGain(dataset, outputs, targets):
    classes = set(targets)
    c = Counter(targets)
    co = Counter(outputs)
    t = Counter(zip(outputs, targets))
    n = float(len(targets))
    print c
    print "N = ", n,
    print classes
    ENTotal = sum ([ - c[cls] / float(n) * log2(c[cls] / float(n) ) for cls in classes])
    print ENTotal
    SENk = 0
    for k in classes:
        ENk = 0
        nk = float(co[k])
        for cls in classes:
            v = t[(k, cls)]

            if v != 0:
                ENk += -(v / nk) * (log2(v / nk))
        w = nk / n
        SENk += w * ENk
    print "ENTotal = ", ENTotal
    print "ENk = " , SENk
    NIG = 1
    if ENTotal != 0:
        NIG = (ENTotal - SENk) / ENTotal
    print "NIG = ", NIG
    return NIG


def main():
    print "KD-Mean algorithm"
    random.seed(5)
    # dataset = random.rand(1000, 2)
    dataset = loadtxt('C:\\Users\\bbphuc\\Desktop\\UCI_Handwriten\\pendigits.tra', delimiter=',')
    targets = dataset[:,-1]
    dataset = dataset[:,:-1]
    # plt.plot(dataset[:,0], dataset[:,1], 'o')
    # plt.show()
    outputs = kmeans(dataset, 10)
    calcInformationGain(dataset, outputs, targets)

if __name__ == '__main__':
    main()
