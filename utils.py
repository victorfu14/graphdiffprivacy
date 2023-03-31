import numpy as np

def partition(l, num_chunks):
    arr = np.array(l, dtype=object)
    return np.array_split(arr, num_chunks)

def partition_binary(l):
    if len(l) == 0:
        return []
    if len(l) == 1:
        return [l]
    index = 2 ** int(np.log2(len(l)))
    return [l[:index]] + partition_binary(l[index:])

def update_psum_naive(arr, global_sen, epsilon, psum_size = 10):
    return np.array([sub_arr.sum() + np.random.laplace(scale = global_sen / epsilon) 
                        for sub_arr in partition(arr, psum_size)])

    