import os
import zmq
import json
import time
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from multiprocessing import Process
from azint import AzimuthalIntegrator
from bitshuffle import decompress_lz4

INTERNAL_PORT = 5550

class Task(object):
    """
    Class representing a type of work on a frame, like integration or
    downsampling.
    """
    def __init__(self, name):
        self.name = name

    def make_extras(self):
        """
        Generate extra data, for example the angular axes for
        integrated data. Called once at the beginning.
        """
        return {}

    def perform(self, frame):
        raise NotImplementedError

class Downsampling(Task):
    """
    Uses stride_tricks to quickly bin pixels in n-by-n groups. Faster
    than a plain c-loop for n>8 or so, only a little slower for n<8.
    """
    def __init__(self, name, n):
        super().__init__(name)
        self.n = n

    def perform(self, frame):
        n = self.n
        strided = as_strided(frame,
            shape=(frame.shape[0]//n, frame.shape[1]//n, n, n),
            strides=((frame.strides[0]*n, frame.strides[1]*n)+frame.strides))
        return strided.sum(axis=-1).sum(axis=-1)

class Integration(Task):
    """
    Uses the fast azint module for cake integration. The parameters
    are just a list of *args that is passed to azint.AzimuthalIntegrator.
    """
    def __init__(self, name, ai):
        super().__init__(name)
        self.ai = ai

    def perform(self, frame):
        return self.ai.integrate(frame)

    def make_extras(self):
        return {'q': self.ai.q, 'phi': self.ai.phi}

class Worker(Process):
    """
    General worker which can set up for various sorts of work.
    """
    def __init__(self, worker_id, nworkers, tasks):
        super().__init__()
        self.worker_id = worker_id
        self.nworkers = nworkers
        self.tasks = tasks
        print('Constucting worker %u out of %u'%(worker_id, nworkers))

class ZmqWorker(Worker):
    def __init__(self, *args, pullhost='tcp://p-daq-cn-2', pullport=20001):
        super().__init__(*args)
        self.pullhost = pullhost
        self.pullport = pullport

    def run(self):
        context = zmq.Context()
        pull_sock = context.socket(zmq.PULL)
        pull_sock.connect('%s:%u'%(self.pullhost, self.pullport))
        push_sock = context.socket(zmq.PUSH)
        push_sock.connect('tcp://localhost:%u'%INTERNAL_PORT)
        while True:
            parts = pull_sock.recv_multipart(copy=False)
            header = json.loads(parts[0].bytes)
            print("header: ", header)
            if header['htype'] == 'image':
                img = decompress_lz4(parts[1].buffer, header['shape'], np.dtype(header['type']))
                print("after decomp.")
                for itask, task in enumerate(self.tasks):
                    res = task.perform(img)
                    header['type'] = str(res.dtype)
                    header['shape'] = res.shape
                    header['compression'] = 'none'
                    header['name'] = task.name
                    push_sock.send_json(header, flags=zmq.SNDMORE)
                    flag = 0 if (itask == len(self.tasks) - 1) else zmq.SNDMORE
                    push_sock.send(res, flag)
                print("for loop done.")
            else:
                push_sock.send_json(header)

class Hdf5Worker(Worker):
    def __init__(self, *args, filename=''):
        super().__init__(*args)
        self.filename = filename

    def run(self):
        context = zmq.Context()
        push_sock = context.socket(zmq.PUSH)
        push_sock.connect('tcp://localhost:%u'%INTERNAL_PORT)
        fh = h5py.File(self.filename, 'r')
        dset = fh['/entry/measurement/pilatus/frames']
        nimages = len(dset)
        if self.worker_id == 0:
            hdr = {'filename': self.filename,
                   'htype': 'header',
                   'msg_number': -1}
            # possibly a race condition if this process becomes delayed, would
            # have wanted a barrier here to make sure the header comes first.
            # not likely though.
            push_sock.send_json(hdr)
        for i in range(self.worker_id, nimages, self.nworkers):
            img = dset[i]
            header = {'msg_number': i,
                      'compression': 'none',
                      'htype': 'image'}
            for itask, task in enumerate(self.tasks):
                res = task.perform(img)
                header['type'] = str(res.dtype)
                header['shape'] = res.shape
                header['name'] = task.name
                push_sock.send_json(header, flags=zmq.SNDMORE)
                flag = 0 if (itask == len(self.tasks) - 1) else zmq.SNDMORE
                push_sock.send(res, flag)

            if i == nimages - 1:
                closing = {'htype': 'series_end',
                           'msg_number': nimages,}
                push_sock.send_json(closing)

def ordered_recv(sock):
    cache = {}
    next_msg_number = 0
    while True:
        parts = sock.recv_multipart()
        header = json.loads(parts[0])
        msg_number = header['msg_number']
        if header['htype'] == 'header':
            next_msg_number = msg_number
        if msg_number == next_msg_number:
            yield msg_number, parts
            next_msg_number += 1
            while next_msg_number in cache:
                data = cache.pop(next_msg_number)
                yield next_msg_number, data
                next_msg_number += 1
        else:
            cache[msg_number] = parts
    

def collector(tasks, disposable=False):
    context = zmq.Context()
    pull_sock = context.socket(zmq.PULL)
    pull_sock.bind('tcp://*:%u'%INTERNAL_PORT)
    fh = None
    frames_total = 0
    frames_since_print = 0
    next_print = int(time.time() // 1)
    for index, parts in ordered_recv(pull_sock):
        frames_total += 1
        frames_since_print += 1
        if int(time.time() // 1) >= next_print:
            print('%u new, %u total'%(frames_since_print, frames_total))
            frames_since_print = 0
            next_print = int(time.time() // 1) + 1
        header = json.loads(parts[0])
        htype = header['htype']
        if htype == 'image':
            for task in tasks:
                header = json.loads(parts.pop(0))
                payload = parts.pop(0)
                res = np.frombuffer(payload, header['type']).reshape(header['shape'])
                dsetname = header['name']
                if fh:
                    dset = fh.get(dsetname)
                    if not dset:
                        print("create dataset")
                        for k, v in task.make_extras().items():
                            if v is not None:
                                fh.create_dataset(k, data=v)
                        dset = fh.create_dataset(dsetname, dtype=header['type'],
                                                   shape=(0, *res.shape), 
                                                   maxshape=(None, *res.shape),
                                                   chunks=(1, *res.shape))
                    n = dset.shape[0]
                    dset.resize(n+1, axis=0)
                    dset[n] = res

        elif htype == 'header':
            filename = header['filename']
            if filename:
                if fh:
                    fh.close()                
                path, fname = os.path.split(filename)
                output_folder = path.replace('raw', 'process/frameprocessing')
                output_file = os.path.join(output_folder, fname)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                fh = h5py.File(output_file, 'a')
            else:
                fh = None
                
        elif htype == 'series_end':
            print('end')
            if fh:
                fh.close()
            if disposable:
                return

if __name__ == '__main__':

    # parameters and shared memory for radial integration
    poni_file = '/data/staff/nanomax/commissioning_2021-1/20210427/process/temp.poni'
    pixel_size = 75.0e-6
    n_splitting = 4
    mask = np.load('/data/visitors/nanomax/common/masks/eiger/20210427.npy')
    phi_bins = np.linspace(-np.pi, np.pi, 701)
    bins = [2000, phi_bins] # if you want cake integration
    bins = [2000,]
    ai = AzimuthalIntegrator(poni_file, mask.shape, pixel_size, n_splitting, bins, mask=mask, solid_angle=True)

    # set up a list of tasks for each process to do
    tasks = []
    tasks.append(Integration(name='I', ai=ai))
    #tasks.append(Downsampling(name='binned', n=5))
    #tasks.append(Downsampling(name='binned_a_lot', n=100))

    nworkers = 12
    procs = []
    for i in range(nworkers):
        #p = ZmqWorker(i, nworkers, tasks, pullhost='tcp://p-daq-cn-1', pullport=20001)
        p = Hdf5Worker(i, nworkers, tasks, filename='/data/staff/nanomax/commissioning_2021-1/20210427/raw/sample/scan_000037_eiger.hdf5')
        p.start()
        procs.append(p)

    collector(tasks, disposable=True) # set this True for offline hdf5 processing
        
    for i in range(nworkers):
        procs[i].join()

#    ai.close()
