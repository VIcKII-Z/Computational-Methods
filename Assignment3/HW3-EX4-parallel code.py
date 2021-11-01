import math
import multiprocessing
import random
import time
import numpy as np


def merge_sort(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(merge_sort(data[:split]))
        right = iter(merge_sort(data[split:]))
        result = []
        # note: this takes the top items off the left and right piles 
        left_top = next(left)
        right_top = next(right)
        while True:
            if left_top < right_top: 
                result.append(left_top) 
                try:
                    left_top = next(left)
                except StopIteration:
                    # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right) 
            else:
                result.append(right_top) 
                try:
                    right_top = next(right)
                except StopIteration:
                    # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left)
                
                
def merge(*args):

    left, right = args[0] if len(args) == 1 else args
    left_length, right_length = len(left), len(right)
    left_index, right_index = 0, 0
    merged = []
    while left_index < left_length and right_index < right_length:
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    if left_index == left_length:
        merged.extend(right[right_index:])
    else:
        merged.extend(left[left_index:])
    return merged

def merge_sort_parallel(data):

    processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=processes)#4

    mid = len(data) // 2
    data = [data[:mid], data[mid:]]
    data = pool.map(merge_sort, data)

    data = merge(data[0], data[1])
#     while len(data) > 1:
#         # If the number of partitions remaining is odd, we pop off the
#         # last one and append it back after one iteration of this loop,
#         # since we're only interested in pairs of partitions to merge.
#         extra = data.pop() if len(data) % 2 == 1 else None
#         data = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]
#         data = pool.map(merge, data) + ([extra] if extra else [])
    return data


if __name__ == "__main__":
    res = {'merge_sort': [], 'merge_sort_parallel': []}
    
    for size in range(10000, 110000, 10000):
        res['merge_sort'].append(0)
        res['merge_sort_parallel'].append(0)
        for trail in range(10):
            data_unsorted = [random.randint(0, size) for _ in range(size)]  
            
            for sort in merge_sort, merge_sort_parallel:
                
                start = time.time()
                data_sorted = sort(data_unsorted)
                end = time.time() - start
                res[sort.__name__][-1] += end
               
                assert sorted(data_unsorted) == data_sorted, 'merge sort not validated'
        
            res['merge_sort'][-1] /= 20
            res['merge_sort_parallel'][-1] /= 20



    print(res)