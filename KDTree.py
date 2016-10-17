from random import seed, random
from time import clock
from collections import namedtuple
from pprint import pformat
from operator import itemgetter
import os
import numpy as np

class Node(namedtuple('Node', 'location left_child right_child leaf_bucket')):
    def __repr__(self):
        return pformat(tuple(self))


def kdtree(point_list, depth=0, bucket_threshold=1):
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
    axis = depth % k
    # Sort point list and choose median
    point_list.sort(key=itemgetter(axis))
    # FIXME: Use mean to split diamension can cause less computing
    median = len(point_list) // 2
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )

def load_pendigits(url):
    dataset = np.loadtxt(url, delimiter=',')
    return dataset

def main():
    """Example """
    point_list = [(2, 3), (2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    n = len(point_list)
    k = len(point_list[0]) # Can cause error
    bucket_threshold =  n / (10 * k) # FIXME: approriate 10 bucket for a cluster
    tree = kdtree(point_list)
    print tree

if __name__ == '__main__':
    main()