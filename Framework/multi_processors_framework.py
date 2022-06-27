import time
import multiprocessing
import os
import numpy as np

orig_path = r'/media/tokuworkstation1/New Volume/Dataset/AREDS/BP Training/AREDS Original Good Quality'

number_of_multiprocessing_thread = 20

list_files = os.listdir(orig_path)
parts = np.array_split(list_files, number_of_multiprocessing_thread)


def basic_func(x):
    n = 0
    for filename in parts[x]:
        # Do something here
        print(filename)
        print('process ' + str(n) + ' out of' + str(parts[x].size))
        n = n + 1


def multiprocessing_func(core_num):
    print('{} squared results in a/an {} number'.format(core_num, basic_func(core_num)))


if __name__ == '__main__':
    starttime = time.time()
    pool = multiprocessing.Pool()
    pool.map(multiprocessing_func, range(0, number_of_multiprocessing_thread))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))
