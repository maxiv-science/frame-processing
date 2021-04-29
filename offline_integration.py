"""
Utility script for offline radial integration of streamed data.

Need to make this a proper argparse with defaults
"""

from pipeline import *
import sys

if len(sys.argv) < 6:
    print('Usage:')
    print('online_integration.py <datafile> <poni> <psize> <mask> <q_bins> <phi_bins>')
    print('\n Cake integration and more settings are avilable but hard coded for now.')
    exit(0)

# parameters and shared memory for radial integration
data_file = sys.argv[1]
poni_file = sys.argv[2]
pixel_size = float(sys.argv[3])
maskfile = sys.argv[4]
q_bins = int(sys.argv[5])
phi_bins = int(sys.argv[6])

# hard coded parameters
n_splitting = 4
mask = np.load(maskfile)
if phi_bins == 1:
    bins = [q_bins,]
else:
    phi_bins = np.linspace(-np.pi, np.pi, phi_bins+1)
    bins = [q_bins, phi_bins] # if you want cake integration
ai = AzimuthalIntegrator(poni_file, mask.shape, pixel_size, n_splitting, bins, mask=mask, solid_angle=True)

# set up a list of tasks for each process to do
tasks = [Integration(name='I', ai=ai),]

nworkers = 12
procs = []
for i in range(nworkers):
    p = Hdf5Worker(i, nworkers, tasks, filename=data_file)
    p.start()
    procs.append(p)

collector(tasks, disposable=True) # set this True for offline hdf5 processing
    
for i in range(nworkers):
    procs[i].join()
