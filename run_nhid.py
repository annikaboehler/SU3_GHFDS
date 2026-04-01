from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import jax
jax.distributed.initialize()

print(f"Rank={rank}: Total number of GPUs: {jax.device_count()}, devices: {jax.devices()}")
print(f"Rank={rank}: Local number of GPUs: {jax.local_device_count()}, devices: {jax.local_devices()}", flush=True)
# wait for all processes to show their devices
comm.Barrier()


import sys
sys.path.insert(1, 'src/')
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


from hiddenfermions_su3_sym import *
from SU3Exchange import *
#from helper import *


parser = argparse.ArgumentParser()
parser.add_argument("-Nx" , "--Nx"   , type=int,  default = 4 , help="length in x dir")
parser.add_argument("-Ny" , "--Ny"   , type=int,  default = 4 , help="length in y dir")
parser.add_argument("-J"  , "--J"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-t"  , "--t"    , type=float,default = 3. , help="hopping amplitude")
parser.add_argument("-Ne"  , "--n_elecs"    , type=int,default = 10 , help="number of electrons")
parser.add_argument("-b1"  , "--b1"    , type=int,default = 0 , help="boundary for x-dir (0:periodic, 1:open)")
parser.add_argument("-b2"  , "--b2"    , type=int,default = 0 , help="boundary for y-dir (0:periodic, 1:open)")
parser.add_argument("-init"  , "--MFinit"    , type=str, default = "Fermi_ED" , help="initialization for MF, careful: Fermi and Fermi_obc initialize square lattice, Fermi_ED initializes triangular lattice, choose one of: Fermi_ED, Fermi, random, Fermi_obc")
parser.add_argument("-f"  , "--features"    , type=int, default = 32 , help="number of features for transformer / FFNN / CNN")
parser.add_argument("-l"  , "--layers"    , type=int, default = 1 , help="number of layers")
parser.add_argument("-n_hid", "--n_hid", type=int, default=20, help="number of hidden fermions")
parser.add_argument("-mu", "--mu", type=float, default=1., help="chemical potential for pinning fields")
parser.add_argument("-m3", "--m3", type=int, default=0, help="rotational m3 quantum number of triangular lattice rotation symmetry")
parser.add_argument("-sym", "--symmetry", type=str, default="no symmetry", help="symmetry averaging of variational state, choose: lattice rotation or spin rotation")
parser.add_argument("-param_type", "--param_type", type=str, default='real', help='dtype of parameters')
parser.add_argument("-load_state", "--load_state", type=str, default='false', help='start from previous state')

args = parser.parse_args()
L1      = args.Nx
L2      = args.Ny
n_elecs = args.n_elecs
J      = args.J
t       = args.t
b1      = args.b1
b2      = args.b2
mu      = args.mu
m3      = args.m3
print("params: Jz=", J, "t=", t, "Lx=", L1, "Lt=", L2, "bounds=", b1, b2, "mu=", mu, "m3=", m3)
MFinitialization = args.MFinit
parameter_type = args.param_type

# more parameters for the physical system
pbc     = [{0: True, 1:False}[b1],{0: True, 1:False}[b2]]
N_sites = L1*L2
N_r     = (n_elecs+2)//3
N_g     = (n_elecs+1)//3
N_b    = n_elecs//3

double_occupancy = False
symmetry=args.symmetry
if args.load_state=='true':
    load=True
elif args.load_state=='false':
    load = False

# network parameters and sampling
lr               = 0.02
n_chains         = 3*1024
n_samples        = 3*1024
n_steps          = 1500
n_hid            = args.n_hid
features         = args.features
layers           = args.layers
n_modes          = 3*L1*L2
cs               = 3*1024


# --------------- define the network -------------------
boundary_conditions = 'pbc' if pbc[0] else 'obc'
graph = nk.graph.Triangular(extent=[L1,L2],pbc=pbc)
hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1, n_fermions_per_spin = (N_r, N_g, N_b))
print(hi.size)

n_params = (n_hid+n_modes)*(n_elecs+n_hid)+features*n_hid*(n_hid+n_elecs)+features*(n_modes+1)+1
print("nparams=", n_params)

ngpu=jax.device_count()

#number of parameters has to be divisible by number of gpus for parallelization, if not add dummy parameters b
if n_params%ngpu!=0:
    need_b = True
    number_b = ngpu-n_params%ngpu
else:
    need_b = False
    number_b = 0


if parameter_type=='real':
    dtype = jnp.float64
    print("using real params")
elif parameter_type=='complex':
    dtype=jnp.complex128
    print("using complex params")

filename = f"results/{L1}x{L2}/states/triangular_energy_{symmetry}_m3={m3}_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}_tri_dtype_{dtype}"
ma=HiddenFermion(n_elecs,"FFNN",n_hid,L1,L2,layers=layers,features=features,double_occupancy_bool=double_occupancy,MFinit=MFinitialization, hilbert=hi, need_b=need_b, number_b=number_b, dtype=dtype, symmetry=symmetry, graph=graph, m3=m3)

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

r_left = np.arange(0,L1,3)
b_left = np.arange(1,L1,3)
g_left = np.arange(2,L1,3)

L = N_sites
j = (L2-1)%3
r_ind = np.append(r_left, L-L1+np.arange(j, L1, 3))
b_ind = np.append(b_left, L-L1+np.arange((j+1)%3, L1, 3))
g_ind = np.append(g_left, L-L1+np.arange((j+2)%3, L1, 3))
print(r_ind, b_ind, g_ind)

for i in r_ind:
    #red pinning filed
    ha -= mu*nc(hi, i, r)
for i in b_ind:
    #blue pinning filed
    ha -= mu*nc(hi, i, b)
for i in g_ind:
    #green pinning field
    ha -= mu*nc(hi, i, g)

# ---------- define sampler ------------------------
if double_occupancy:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=ExchangeRule(graph=graph))
else:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=tJExchangeRule(graph=graph))

vstate = nk.vqs.MCState(sa, ma, n_discard_per_chain=32, n_samples=n_samples, chunk_size=cs) #defines the variational state object
if load:
    print("loading state from:", filename)
    vars = nkx.vqs.variables_from_file(filename+".mpack", vstate.variables)
    # update the variables of vstate with the loaded data.
    vstate.variables = vars
    filename = filename+"_2"
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# thermalize samples
for i in range(5):
  vstate.sample()
print("state thermalized")

# -------------- start the training ---------------
initial_energy, initial_grad = vstate.expect_and_grad(ha)

# print(initial_energy.mean)
# print(initial_grad)
# if np.any(np.isnan(np.real(initial_grad['a']))):
#   raise("Initial Energy is nan, stopping optimization")
# if np.any(np.isnan(np.real(initial_grad['hidden_0']['kernel']))):
#   raise("Initial Energy is nan, stopping optimization")
# if np.any(np.isnan(np.real(initial_grad['hidden_0']['bias']))):
#   raise("Initial Energy is nan, stopping optimization")
# if np.any(np.isnan(np.real(initial_grad['output']['kernel']))):
#   raise("Initial Energy is nan, stopping optimization")
# if np.any(np.isnan(np.real(initial_grad['output']['bias']))):
#   raise("Initial Energy is nan, stopping optimization")
# if np.any(np.isnan(np.real(initial_grad['orbitals']['orbitals_hf']))):
#   raise("Initial Energy is nan, stopping optimization")
# if np.any(np.isnan(np.real(initial_grad['orbitals']['orbitals_mf']))):
#   raise("Initial Energy is nan, stopping optimization")
# if np.isnan(np.real(initial_energy.mean)):
#   raise("Initial Energy is nan, stopping optimization")


schedule = optax.linear_schedule(init_value=0.1, end_value=0.001, transition_steps=n_steps)
op = nk.optimizer.Sgd(learning_rate=schedule)
schedule_ds = optax.linear_schedule(init_value=0.1, end_value=0.00001, transition_steps=n_steps)
gs = nkx.driver.VMC_SRt(ha, op, variational_state=vstate, jacobian_mode='complex', diag_shift=schedule_ds)

print("starting optimization ...")
gs.run(n_iter=n_steps, out=filename)

final_samples = vstate.samples

np.save(f"results/{L1}x{L2}/samples/triangular_energy_{symmetry}_m3={m3}_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}_tri_dtype_{dtype}.npy", np.asarray(final_samples))
