from numpy import *
import random as rnd
import matplotlib.pyplot as plt
from collections import Counter
from KDTree import KDTree2centroids
MAX_ITERATIONS = 20


def getRandomCentroids(dataset, k, type='Forgy'):
    """ Return a list of Centroids
        Attribute:
            dataset: array of points
            k : number of cluster = number of centroids
            type: 'Forgy' | 'Random Partition'
    """
    numFeatures = 1
    numData = shape(dataset)[0]
    if ndim(dataset) > 1:
        numFeatures = shape(dataset)[1]
    # Init result array
    res = []
    # Assign k sample as centroids
    if type == 'Forgy':
        res = array(rnd.sample(dataset, k))
    else: # Random Partition
        labels = random.randint(0, k, len(dataset))
        # in case a cluster has no points
        check = in1d(range(k), labels)
        if False in check:
            res = getRandomCentroids(dataset, k, type)
        else:
            res = getCentroids(dataset, labels, k, [])
    print "Init centroids with ", type
    print res
    print "==========="
    return res

def getCentroids(dataset, labels, k, oldCentroids):
    newCentroids = []
    for xcluster in range(k):
        cluster = [i for i, j in zip(dataset, labels) if j == xcluster]
        # in case a cluster is empty
        if len(cluster) > 0:
            newCentroids.append(mean(cluster, axis=0))
        else: # cluster is empty
            newCentroids.append(oldCentroids[xcluster]) # assign new = oldcentroids for empty cluster
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


def kmeans(dataset, k, initmethod):
    # Initialize centroids randomly ( Forgy approch)
    numFeatures = 1
    if ndim(dataset) > 1:
        numFeatures = shape(dataset)[1]
    #centroids = getRandomCentroids(dataset, k)
    centroids = None
    if initmethod == 'kdtree':
        centroids = KDTree2centroids(dataset, k)
    else:
        centroids = getRandomCentroids(dataset, k, initmethod)
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
        centroids = getCentroids(dataset, labels, k, oldCentroids)
    # plt.show()
    figure2 = plt.figure(str(initmethod))
    ax2 = figure2.add_subplot(111)
    color = ['r', 'b', 'g', 'c'] * 3
    for i in range(k):  # For each
        # print labels
        cluster = []
        for j in range(len(dataset)):
            if labels[j] == i:
                cluster.append(dataset[j])
        if len(cluster) > 0:
            cluster = array(cluster)
            # print cluster
            plt.plot(cluster[:, 0], cluster[:, 1], color[i % 4] + 'o')
    plt.plot(centroids[:, 0], centroids[:, 1], 'k^')
    print "Iteraction of " + initmethod + ": "+str(iterations)
    print "==========="
    return labels

def generateGaussianData(n, nCentroids):
    """ Generate dataset randomly with Gaussian distribution
        Generate noise and add into dataset
        Return dataset
        Attribute:
            n : number of points in each cluster
            nCentroids: number of cluster (k in K-Mean)
    """
    random.seed()
    # Random n centroids
    means = random.uniform(0, 10,(nCentroids,3))
    # Choose R depending on minimum distance of centroids
    r1 = float('inf');
    for i in xrange(nCentroids - 1):
        for j in range(i + 1,nCentroids):
            if r1 > distance(means[i],means[j]):
                r1 = distance(means[i],means[j])
    scale = 0.2
    r1 = r1*scale

    # generate gaussian data arround these centroids
    s = 0.05
    r2 = random.uniform(0.2 * s * sqrt(n), s * sqrt(n))
    r = min(r1, r2)
    # 
    cov = [[r,0],[0, r]]
    cluster = [(random.multivariate_normal(means[i], cov, n), i) for i in range(len(means))]
    # Get dataset and targets from dataset
    point, tar = [point for point, tar in cluster], [tar for point, tar in cluster]
    # Draw 
    fg = plt.figure('Clear Data')
    plt.axis('equal')
    ax = fg.add_subplot(111)
    color = ['r', 'b', 'g', 'c'] * 3
    for i, c in enumerate(point):
        plt.plot(c[:, 0], c[:, 1], color[i] + 'o')

    #clear_data = array([c[j] for c in point for j in range(len(c))])
    maxi, mini = max(array(point).flatten()) , min(array(point).flatten())
    # Add noise to dataset
    noise = random.uniform(mini, maxi, (n // 10, 2))
    tar_noise = getLabels(noise, means)   #random.randint(0, nCentroids, n // 10)
    point = array([c[j] for c in point for j in range(n)])
    #print shape(noise)
    #print shape(point)
    tar = [i for i in range(nCentroids) for j in range(n)]

    return concatenate((noise,point), axis = 0), concatenate((tar_noise, array(tar)), axis = 0)

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
    #dataset = loadtxt('pendigits.tra', delimiter=',')
    kcluster = 4
    dataset, labels = generateGaussianData(300, kcluster)
    # print "Label", labels
    figure2 = plt.figure("Target")
    ax2 = figure2.add_subplot(111)
    color = ['r', 'b', 'g', 'c'] * 3
    for i in range(kcluster):  # For each
        # print labels
        cluster = []
        for j in range(len(dataset)):
            if labels[j] == i:
                cluster.append(dataset[j])
        if len(cluster) > 0:
            cluster = array(cluster)
            # print cluster
            plt.plot(cluster[:, 0], cluster[:, 1], color[i % 4] + 'o')
    #plt.plot(centroids[:, 0], centroids[:, 1], 'k^')

    #dataset = loadtxt('test.txt', delimiter=',')
    #dataset = [(2, 3), (3, 2), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2), (8,8), (10,3),(1,1),(5,9), (9,3)]
    #targets = dataset[:,-1]
    #dataset = dataset[:,:-1]
    # plt.plot(dataset[:,0], dataset[:,1], 'o')
    # plt.show()
    outputs = kmeans(dataset, kcluster, 'Forgy')
    output1 = kmeans(dataset, kcluster, 'Random Partition')
    output2 = kmeans(dataset, kcluster, 'kdtree')
    print 'Forgy'
    calcInformationGain(dataset, outputs, labels)
    print 'randompar'
    calcInformationGain(dataset, output1, labels)
    print 'kdtree'
    calcInformationGain(dataset, output2, labels)

    plt.show()

if __name__ == '__main__':
    main()
