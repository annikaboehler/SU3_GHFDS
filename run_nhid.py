import sys
sys.path.insert(1, '/project/th-scratch/a/Annika.Boehler/PhD/SU3/NQS/src/')
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

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc


from hiddenfermions_su3_sym_single import *
from SU3Exchange_sym import *


parser = argparse.ArgumentParser()
parser.add_argument("-Nx" , "--Nx"   , type=int,  default = 4 , help="length in x dir")
parser.add_argument("-Ny" , "--Ny"   , type=int,  default = 4 , help="length in y dir")
parser.add_argument("-J"  , "--J"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-t"  , "--t"    , type=float,default = 3. , help="hopping amplitude")
parser.add_argument("-Ne"  , "--n_elecs"    , type=int,default = 10 , help="number of electrons")
parser.add_argument("-b1"  , "--b1"    , type=int,default = 0 , help="boundary for x-dir (0:periodic, 1:open)")
parser.add_argument("-b2"  , "--b2"    , type=int,default = 0 , help="boundary for y-dir (0:periodic, 1:open)")
parser.add_argument("-init"  , "--MFinit"    , type=str, default = "Fermi" , help="initialization for MF")
parser.add_argument("-f"  , "--features"    , type=int, default = 32 , help="number of features for transformer / FFNN / CNN")
parser.add_argument("-l"  , "--layers"    , type=int, default = 1 , help="number of layers")
parser.add_argument("-n_hid", "--n_hid", type=int, default=20, help="number of hidden fermions")
parser.add_argument("-mu", "--mu", type=float, default=1., help="chemical potential for pinning fields")
load = False

args = parser.parse_args()
L1      = args.Nx
L2      = args.Ny
n_elecs = args.n_elecs
J      = args.J
t       = args.t
b1      = args.b1
b2      = args.b2
mu      = args.mu
print("params: Jz=", J, "t=", t, "Lx=", L1, "Lt=", L2, "bounds=", b1, b2, "mu=", mu)
MFinitialization = args.MFinit


# more parameters for the physical system
pbc     = [{0: True, 1:False}[b1],{0: True, 1:False}[b2]]
N_sites = L1*L2
N_r     = (n_elecs+2)//3
N_g     = (n_elecs+1)//3
N_b    = n_elecs//3

double_occupancy = False

# network parameters and sampling
lr               = 0.02
n_chains         = 3*2048
n_samples        = 3*2048
n_steps          = 1500
n_hid            = args.n_hid
features         = args.features
layers           = args.layers
n_modes          = 3*L1*L2
cs               = 2048


# --------------- define the network -------------------
boundary_conditions = 'pbc' if pbc[0] else 'obc'
graph = nk.graph.Triangular(extent=[L1,L2],pbc=pbc)
hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1, n_fermions_per_spin = (N_r, N_g, N_b))

n_params = (n_hid+n_modes)*(n_elecs+n_hid)+features*n_hid*(n_hid+n_elecs)+features*(n_modes+1)+1
print("nparams=", n_params)

#nparams/n_gpus must be an integer, add dummy parameter b
if n_params%2==1:
    need_b = True
else:
    need_b = False

filename = f"results/triangular_energy_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}"
ma=HiddenFermion(n_elecs,"FFNN",n_hid,L1,L2,layers=layers,features=features,double_occupancy_bool=double_occupancy,MFinit=MFinitialization, hilbert=hi, need_b=need_b, dtype=jnp.float64)

# ------------- define Hamiltonian ------------------------
r, g, b = +2, 0, -2
ha = 0.0
for sz in (r, g, b):
    for u, v in graph.edges():
        ha += -t*(cdag(hi, u, sz) * c(hi, v, sz)  + cdag(hi, v, sz) * c(hi, u, sz))

for sz1 in (r,g,b):
    for sz2 in (r,g,b):
        for u,v in graph.edges():
            ha += J/2*(cdag(hi, u, sz1)*c(hi, u, sz2)*cdag(hi, v, sz2)*c(hi, v, sz1))
for u,v in graph.edges():
    ha -= J/2*(nc(hi, u, r)+nc(hi, u, g)+nc(hi, u, b))*(nc(hi, v, r)+nc(hi, v, g)+nc(hi, v, b))


# ---------- define sampler ------------------------
if double_occupancy:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=ExchangeRule(graph=graph))
else:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=tJExchangeRule(graph=graph))


vstate = nk.vqs.MCState(sa, ma, n_discard_per_chain=8, n_samples=n_samples, chunk_size=cs) 

if load:
    vars = nkx.vqs.variables_from_file(filename+".mpack", vstate.variables)
    # update the variables of vstate with the loaded data.
    vstate.variables = vars
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# thermalize samples
for i in range(5):
  vstate.sample()

# -------------- start the training ---------------
schedule = optax.linear_schedule(init_value=0.05, end_value=0.001, transition_steps=n_steps)
op = nk.optimizer.Sgd(learning_rate=schedule)
schedule_ds = optax.linear_schedule(init_value=0.01, end_value=0.000001, transition_steps=n_steps)
gs = nkx.driver.VMC_SRt(ha, op, diag_shift=schedule_ds, variational_state=vstate)
print("starting optimization ...")
gs.run(n_iter=n_steps, out=filename)
