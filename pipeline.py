import os
import zmq
import json
import time
import h5py
import numpy as np
from multiprocessing import Process
from azint import AzimuthalIntegrator
from bitshuffle import decompress_lz4
import fabio

class Worker(Process):
    """
    General worker which can set up for various sorts of work.
    """
    def __init__(self, worker_id, nworkers,
                       do_integration=False, integration_params=None,
                       do_downsampling=False, downsampling_params=None):
        super().__init__()
        self.worker_id = worker_id
        self.nworkers = nworkers

        # choose what work to do - this is a bit stupid because making the ai
        # object takes time, and in the constructor it gets done sequentially.
        if do_integration:
            ai = AzimuthalIntegrator(*integration_params, create=False)
            self.process = lambda img: ai.integrate(img)
        print('Constucting worker %u out of %u'%(worker_id, nworkers))

class ZmqWorker(Worker):
    def __init__(self, *args, pullhost='tcp://p-daq-cn-2', pullport=20001, pushport=5550, **kwargs):
        super().__init__(*args, **kwargs)
        self.pullhost = pullhost
        self.pullport = pullport
        self.pushport = pushport

    def run(self):
        context = zmq.Context()
        pull_sock = context.socket(zmq.PULL)
        pull_sock.connect('%s:%u'%(self.pullhost, self.pullport))
        push_sock = context.socket(zmq.PUSH)
        push_sock.connect('tcp://localhost:%u'%self.pushport)
        while True:
            parts = pull_sock.recv_multipart(copy=False)
            header = json.loads(parts[0].bytes)
            if header['htype'] == 'image':
                img = decompress_lz4(parts[1].buffer, header['shape'], np.dtype(header['type']))
                res = self.process(img)
                header['type'] = 'float32'
                header['shape'] = res.shape
                header['compression'] = 'none'
                push_sock.send_json(header, flags=zmq.SNDMORE)
                push_sock.send(res)
            else:
                push_sock.send_json(header)

class Hdf5Worker(Worker):
    def __init__(self, *args, filename=None, pushport=5550, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushport = pushport
        self.filename = filename

    def run(self):
        context = zmq.Context()
        push_sock = context.socket(zmq.PUSH)
        push_sock.connect('tcp://localhost:%u'%self.pushport)
        fh = h5py.File(self.filename, 'r')
        dset = fh['/entry/measurement/Eiger/data']
        nimages = len(dset)
        if self.worker_id == 0:
            hdr = {'filename': self.filename,
                   'htype': 'header',
                   'msg_number': -1}
            push_sock.send_json(hdr)
        for i in range(self.worker_id, nimages, self.nworkers):
            img = dset[i]
            res = self.process(img)
            header = {'msg_number': i,
                      'type': 'float32',
                      'shape': res.shape,
                      'compression': 'none',
                      'htype': 'image',}
            push_sock.send_json(header, flags=zmq.SNDMORE)
            push_sock.send(res)
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
    

def collector(radial_bins, phi_bins, disposable=False):
    base_folder = '/data/staff/nanomax/commissioning_2020-2/20201214_integ/process/radial_integration/'
    context = zmq.Context()
    pull_sock = context.socket(zmq.PULL)
    pull_sock.bind('tcp://*:5550')
    fh = None
    dset = None
    for index, parts in ordered_recv(pull_sock):
        header = json.loads(parts[0])
        print(index, header)
        htype = header['htype']
        if htype == 'image':
            res = np.frombuffer(parts[1], header['type']).reshape(header['shape'])
            if fh:
                if not dset:
                    fh.create_dataset('q', data=0.5*(radial_bins[:-1] + radial_bins[1:]))
                    fh.create_dataset('phi', data=0.5*(phi_bins[:-1] + phi_bins[1:]))
                    dset = fh.create_dataset('cake', dtype=np.float32, 
                                               shape=(0, *res.shape), 
                                               maxshape=(None, *res.shape),
                                               chunks=(1, *res.shape))
                n = dset.shape[0]
                dset.resize(n+1, axis=0)
                dset[n] = res

        elif htype == 'header':
            filename = header['filename']
            if filename:
                path, fname = os.path.split(filename)
                output_folder = os.path.join(base_folder, path.split(os.sep)[-1])
                output_file = os.path.join(output_folder, fname)
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                fh = h5py.File(output_file, 'w-')
            else:
                fh = None
            dset = None
                
        elif htype == 'series_end':
            print('end')
            if fh:
                fh.close()
            if disposable:
                return

if __name__ == '__main__':
    poni_file = '/data/visitors/nanomax/20200364/2020120208/process/Detector_calibration/Si_scan2.poni'
    pixel_size = 75.0e-6
    n_splitting = 4
    nworkers = 8
    mask = fabio.open('/data/visitors/nanomax/20200364/2020120208/process/Detector_calibration/mask_scan2.edf').data
    radial_bins = np.linspace(0.0, 38.44, 301)
    phi_bins = np.linspace(-np.pi, np.pi, 701)
    integ_pars = [poni_file, mask.shape, pixel_size, n_splitting, mask, [radial_bins, phi_bins]]
    ai = AzimuthalIntegrator(*integ_pars)

    procs = []
    for i in range(nworkers):
        p = ZmqWorker(i, nworkers, do_integration=True, integration_params=integ_pars)
#        p = Hdf5Worker(i, nworkers, do_integration=True, integration_params=integ_pars,
#            filename='/data/staff/nanomax/commissioning_2020-2/20201214_integ/raw/sample/scan_000022_eiger.hdf5')
        p.start()
        procs.append(p)

    collector(radial_bins, phi_bins)#, disposable=True)
        
    for i in range(nworkers):
        procs[i].join()

    ai.close()
