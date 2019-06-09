import mxnet as mx
import numpy as np

import random
import os
import cv2

class SaliconIter(mx.io.DataIter):
    def __init__(self, fine_path, coarse_path, label_path, batch_size):
        self.fine_path = fine_path
        self.coarse_path = coarse_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.file_list = self.init_file_list()
        self.batch_total = int(len(self.file_list) // self.batch_size)
        self.batch_done = 0

        self._provide_data = zip(['fine', 'coarse'], [(self.batch_size, 3, 80, 80), (self.batch_size, 3, 80, 80)])
        self._provide_label = zip(['label'], [(self.batch_size, 1, 10, 10)])

        self.fine_gen = self.fine_iter(self.file_list)
        self.coarse_gen = self.coarse_iter(self.file_list)
        self.label_gen = self.label_iter(self.file_list)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.file_list = self.init_file_list()
        self.fine_gen = self.fine_iter(self.file_list)
        self.coarse_gen = self.coarse_iter(self.file_list)
        self.label_gen = self.label_iter(self.file_list)
        self.batch_done = 0

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.batch_done < self.batch_total:
            self.batch_done += 1
            data = [mx.nd.array(next(self.fine_gen)), mx.nd.array(next(self.coarse_gen))]
            label = [mx.nd.array(next(self.label_gen))]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration

    def fine_iter(self, lst):
        for i in range(self.batch_total):
            batch = mx.nd.zeros((self.batch_size, 3, 80, 80))
            for j in range(self.batch_size):
                cur_fname = os.path.join(self.fine_path, lst[i*self.batch_size+j] + '.jpg')
                cur_fine = np.swapaxes(cv2.imread(cur_fname), 0, 2).astype(np.float32)
                batch[j] = cur_fine
            yield batch

    def coarse_iter(self, lst):
        for i in range(self.batch_total):
            batch = mx.nd.zeros((self.batch_size, 3, 80, 80))
            for j in range(self.batch_size):
                cur_fname = os.path.join(self.coarse_path, lst[i*self.batch_size+j] + '.jpg')
                cur_coarse = np.swapaxes(cv2.imread(cur_fname), 0, 2).astype(np.float32)
                batch[j] = cur_coarse
            yield batch

    def label_iter(self, lst):
        for i in range(self.batch_total):
            batch = mx.nd.zeros((self.batch_size, 1, 10, 10))
            for j in range(self.batch_size):
                cur_fname = os.path.join(self.label_path, lst[i*self.batch_size+j] + '.jpg')
                cur_label = np.expand_dims(cv2.resize(cv2.imread(cur_fname), (10, 10), interpolation=cv2.INTER_LANCZOS4)[:,:,0].transpose(), axis=0).astype(np.float32)
                batch[j] = cur_label
            yield batch

    def init_file_list(self):
        t = []
        for _,_,fs in os.walk(self.fine_path):
            for f in fs:
                t.append(f.split('.')[0])
        random.shuffle(t)
        return t

    def normalize(self, arr):
        return (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + 1e-10)