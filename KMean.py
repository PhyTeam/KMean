from numpy import *
import random as rnd
import matplotlib.pyplot as plt
MAX_ITERATIONS = 4
def getRandomCentroids(dataset, k):
    numFeatures = 1
    numData = shape(dataset)[0]
    if ndim(dataset) > 1:
        numFeatures = shape(dataset)[1]

    # Assign k sample as centroids
    return  array(rnd.sample(dataset, k))

def getCentroids(dataset, labels, k):
    newCentroids = []
    for xcluster in range(k):
        cluster = [i for i, j in zip(dataset, labels) if j == xcluster]
        newCentroids.append(mean(cluster, axis=0))
    print "{0:-^30}".format("New Centrolds")
    print newCentroids
    return array(newCentroids)


def distance(d1, d2):
    assert len(d1) == len(d2)
    d = len(d1)
    return sqrt(sum([ (i - j) * (i - j) for i,j in zip(d1, d2)]))

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
    print centroids
    figure = plt.figure()
    ax1 = figure.add_subplot(111)

    plt.plot(dataset[:,0], dataset[:,1], 'bo')
    plt.plot(centroids[:,0], centroids[:, 1], 'w.')

    # plt.show()

    # Initialize book keeping vars
    iterations, oldCentroids = 0, None
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids
        oldCentroids = centroids
        iterations += 1

        # Assign labels to each data point based on centroids
        labels = getLabels(dataset, centroids)

        # Assign new centroids base on datapoint label
        figure2 = plt.figure()
        ax2 = figure2.add_subplot(111)


        color = ['r', 'b', 'g', 'c', 'm', 'i', 'k', 'w']
        for i in range(k): # For each
            # print labels
            cluster = []
            for j in range(len(dataset)):
                if labels[j] == i:
                    cluster.append(dataset[j])
            if len(cluster) > 0:
                cluster = array(cluster)
                plt.plot(cluster[:, 0], cluster[:, 1], color[i] + 'o')
        plt.plot(centroids[:, 0], centroids[:,1], 'k^')
        # Calculate new centroids
        centroids = getCentroids(dataset, labels,k)
    plt.show()

def main():
    print "KD-Mean algorithm"
    random.seed(5)
    dataset = random.rand(1000, 2)
    #plt.plot(dataset[:,0], dataset[:,1], 'o')
    #plt.show()
    kmeans(dataset, 5)

if __name__ == '__main__':
    main()