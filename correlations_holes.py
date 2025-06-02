import sys
#sys.path.insert(1, '/project/th-scratch/a/Annika.Boehler/PhD/SU3/NQS/src/')
import argparse
import numpy as np
from jax import numpy as jnp
import netket as nk
import jax
from netket import experimental as nkx
import json
import optax
import os
import flax
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
#import matplotlib.pyplot as plt

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc


from hiddenfermions_su3_sym_single import *
from SU3Exchange_sym import *
#from helper import *


parser = argparse.ArgumentParser()
parser.add_argument("-Nx" , "--Nx"   , type=int,  default = 4 , help="length in x dir")
parser.add_argument("-Ny" , "--Ny"   , type=int,  default = 4 , help="length in y dir")
parser.add_argument("-t"  , "--t"    , type=float,default = 3. , help="hopping amplitude")
parser.add_argument("-J"  , "--J"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-Ne"  , "--n_elecs"    , type=int,default = 10 , help="number of electrons")
parser.add_argument("-b1"  , "--b1"    , type=int,default = 0 , help="boundary for x-dir (0:periodic, 1:open)")
parser.add_argument("-b2"  , "--b2"    , type=int,default = 0 , help="boundary for y-dir (0:periodic, 1:open)")
parser.add_argument("-init"  , "--MFinit"    , type=str, default = "Fermi" , help="initialization for MF")
parser.add_argument("-f"  , "--features"    , type=int, default = 32 , help="number of features for transformer / FFNN / CNN")
parser.add_argument("-l"  , "--layers"    , type=int, default = 1 , help="number of layers")
parser.add_argument("-n_hid", "--n_hid", type=int, default=20, help="number of hidden fermions")
parser.add_argument("-la", "--lattice", type=str, default="triangular", help="lattice type to use, choose square/triangular")
parser.add_argument("-mu", "--mu", type=float, default=0., help="chemical potential for pinning fields")


args = parser.parse_args()
L1      = args.Nx
L2      = args.Ny
n_elecs = args.n_elecs
mu      = args.mu
J       = args.J
t       = args.t
b1      = args.b1
b2      = args.b2
print("params: t=", t, "Lx=", L1, "Lt=", L2, "bounds=", b1, b2)
MFinitialization = args.MFinit
lattice = args.lattice

# more parameters for the physical system
pbc     = [{0: True, 1:False}[b1],{0: True, 1:False}[b2]]
N_sites = L1*L2
N_r     = (n_elecs+2)//3
N_g     = (n_elecs+1)//3
N_b    = n_elecs//3

double_occupancy = False

# network parameters and sampling
lr               = 0.02
n_chains         = 9*2048
n_samples        = 9*2048
n_steps          = 1000
n_hid            = args.n_hid
features         = args.features
layers           = args.layers
n_modes          = 3*L1*L2
cs               = 9*2048
dmax             = L1*L2

# --------------- define the network -------------------
boundary_conditions = 'pbc' if pbc[0] else 'obc'
boundary_conditions_x = 'pbc' if pbc[0] else 'obc'
boundary_conditions_y = 'pbc' if pbc[1] else 'obc'
graph = nk.graph.Triangular([L1,L2],pbc=pbc)
hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1, n_fermions_per_spin = (N_r, N_g, N_b))
print(hi.size)

filename = f'/mnt/data/results/triangular_energy_no_sym_{L1}x{L2}_{boundary_conditions_x}x{boundary_conditions_y}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_Jz={J}_Jp={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}'
ma=HiddenFermion(n_elecs,"FFNN",n_hid,L1,L2,layers=layers,features=features,double_occupancy_bool=double_occupancy,MFinit=MFinitialization, hilbert=hi, dtype=jnp.float64)

# ---------- define sampler ------------------------
if double_occupancy:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=ExchangeRule(graph=graph))
else:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=tJExchangeRule(graph=graph))


vstate = nk.vqs.MCState(sa, ma, n_discard_per_chain=32, n_samples=n_samples, chunk_size=cs) #defines the variational state object
vars = nkx.vqs.variables_from_file(filename+".mpack", vstate.variables)
# update the variables of vstate with the loaded data.
vstate.variables = vars
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# thermalize samples
#for i in range(5):
#  vstate.sample()


def nh(hi,i):
  return 1-nc(hi,i,-2)-nc(hi,i,0)-nc(hi,i,2)

# -------------- calculate expectation values ---------------

samples = np.reshape(vstate.samples, (-1, hi.size))

sr = samples[:,:hi.n_orbitals]
sg = samples[:,hi.n_orbitals:2*hi.n_orbitals]
sb = samples[:,2*hi.n_orbitals:]

samples_rs = sr+sg+sb
_, hole_pos = np.where(samples_rs==0)
hole_pos = hole_pos.reshape(-1, 2)
hole_pos = hole_pos.reshape(-1, 2)

ds = graph.distances()[hole_pos[:,0],hole_pos[:,1]]

np.save(f"/results/hole_corrs/{lattice}_distance_{L1}x{L2}_{boundary_conditions_x}x{boundary_conditions_y}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}.npy", ds)