from random import seed, random
from time import clock
from collections import namedtuple
from pprint import pformat
from operator import itemgetter
import os
import numpy as np
#from KMean import distance

class Node(namedtuple('Node', 'location left_child right_child leaf_bucket')):
    def __repr__(self):
        return pformat(tuple(self))

def distance(d1, d2):
    assert len(d1) == len(d2)
    d = len(d1)
    return np.sqrt(sum([(i - j) * (i - j) for i, j in zip(d1, d2)]))

def kdtree(point_list, depth=0, bucket_threshold= 2):
    try:
        k = len(point_list[0])  # assumes all points have the same diamension
    except IndexError as e:
        return None
    # Check point list has number of point less or equal threshold
    n = len(point_list) 
    if n <= bucket_threshold:
        return Node(
            None,
            None, None,
            point_list
        )
    # Select axis
    # FIXME: Choose longest diamension
    # axis = depth % k
    # Find the longest dimension
    maxwidth = 0
    axis = 0
    for i in xrange(len(point_list[0])):
        arr = [point[i] for point in point_list]
        if (maxwidth < max(arr) - min(arr)):
            axis = i
            maxwidth = max(arr) - min(arr)

    # Sort point list and choose median
    #point_list.sort(key=itemgetter(axis))
    #print len(point_list[0])
    #print 'axis =',axis

    #np.sort(point_list, axis = axis)
    #point_list.view('i8,i8,i8').sort(order=['f1'], axis=0)
    point_list = sorted(point_list, key=lambda a_entry: a_entry[axis]) 
    # FIXME: Use mean to split diamension can cause less computing
    median = len(point_list) // 2
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1,bucket_threshold),
        right_child=kdtree(point_list[median:], depth + 1, bucket_threshold),
        leaf_bucket = None
    )

def getLeafBucket(tree):
    """Return a list of LeafBucket
    tree: Node root
    """
    queue = [tree]
    LeafBucket = []
    while len(queue) > 0:
        node = queue.pop();
        if node.leaf_bucket == None:
            queue.append(node.left_child)
            queue.append(node.right_child)
        else:
            LeafBucket.append(node.leaf_bucket)
    return LeafBucket

def calDensity(lbucket):
    """ Return density of leafbucket - float"""
    dimen = len(lbucket[0])
    # Calculate width of each dimension
    width = []    
    for i in xrange(dimen):
        arr = [point[i] for point in lbucket]
        width.append(max(arr) - min(arr))
    # Calculate volume

    vol = reduce(lambda x,y : x*y, width)
    if vol == 0: 
        density = 0.001
    else:
        density = len(lbucket)/vol
    #print 'LeafBucket'
    #print lbucket
    #print density
    return density
def chooseCentroids(LeafBucket, k):
    """ Return k centroids"""
    lbcentroids = []
    density = []
    # Calculate densities and mean values for each leafbucket
    for lb in LeafBucket:
        density.append(calDensity(lb))
        lbcentroids.append(np.mean(lb, axis=0))

    # Init and choose first centroids
    centroids = []
    first = np.argmax(density)

    centroids.append(lbcentroids[first])
    # Remove chosen lbcentroids
    del density[first]
    del lbcentroids[first]
    #density.remove(density[first])
    #lbcentroids.remove(lbcentroids[first])

    for i in xrange(1,k):
        next = -1
        maxg = 0; # g = mindist*density
        # Choose index of next centroids
        for index, lbc in enumerate(lbcentroids):
            # Calculate min distance from centroids
            mindist = min([distance(c,lbc) for c in centroids])
            if (mindist*density[index] > maxg):
                next = index
                maxg = mindist*density[index]
        centroids.append(lbcentroids[next])
        #print lbcentroids[next]
        del density[next]
        del lbcentroids[next]
        #density.remove(density[next])
        #lbcentroids.remove(lbcentroids[next])
    return centroids


def load_pendigits(url):
    dataset = np.loadtxt(url, delimiter=',')
    return dataset

def KDTree2centroids(point_list, kcluster):
    """Example """
    #point_list = [(2, 3), (3, 2), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2), (8,8), (10,3),(1,1),(5,9), (9,3)]
    n = len(point_list)
    #kcluster = 3
    dim = len(point_list[0]) # Can cause error
    bucket_threshold =  n / (10 * kcluster) # FIXME: approriate 10 bucket for a cluster
    tree = kdtree(point_list)
    # print 'Tree'
    #print tree
    # print '========'
    LeafBucket = getLeafBucket(tree)
    #print LeafBucket
    centroids = chooseCentroids(LeafBucket, kcluster)
    print 'Init Centroids with kdtree'
    print np.array(centroids)
    print '=========='
    return np.array(centroids)

if __name__ == '__main__':
    main()