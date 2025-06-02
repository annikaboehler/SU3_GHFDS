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
parser.add_argument("-Jz"  , "--Jz"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-Jp"  , "--Jp"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-t"  , "--t"    , type=float,default = 3. , help="hopping amplitude")
parser.add_argument("-Ne"  , "--n_elecs"    , type=int,default = 10 , help="number of electrons")
parser.add_argument("-b1"  , "--b1"    , type=int,default = 0 , help="boundary for x-dir (0:periodic, 1:open)")
parser.add_argument("-b2"  , "--b2"    , type=int,default = 0 , help="boundary for y-dir (0:periodic, 1:open)")
parser.add_argument("-init"  , "--MFinit"    , type=str, default = "Fermi" , help="initialization for MF")
parser.add_argument("-f"  , "--features"    , type=int, default = 32 , help="number of features for transformer / FFNN / CNN")
parser.add_argument("-l"  , "--layers"    , type=int, default = 1 , help="number of layers")
parser.add_argument("-n_hid", "--n_hid", type=int, default=20, help="number of hidden fermions")
parser.add_argument("-mu", "--mu", type=float, default=1., help="chemical potential for pinning fields")
parser.add_argument("-la", "--lattice", type=str, default="square", help="lattice type to use, choose square/triangular")

args = parser.parse_args()
L1      = args.Nx
L2      = args.Ny
n_elecs = args.n_elecs
J      = args.Jz
Jp      = args.Jp
t       = args.t
b1      = args.b1
b2      = args.b2
mu      = args.mu
print("params: Jz=", J, "Jp=", Jp, "t=", t, "Lx=", L1, "Lt=", L2, "bounds=", b1, b2, "mu=", mu)
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
n_chains         = 6*2048
n_samples        = 6*2048
n_steps          = 1000
n_hid            = args.n_hid
features         = args.features
layers           = args.layers
n_modes          = 3*L1*L2
cs               = 6*2048
dmax             = L1*L2

# --------------- define the network -------------------
boundary_conditions = 'pbc' if pbc[0] else 'obc'
boundary_conditions_x = 'pbc' if pbc[0] else 'obc'
boundary_conditions_y = 'pbc' if pbc[1] else 'obc'
graph = nk.graph.Triangular([L1,L2],pbc=pbc)
hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1, n_fermions_per_spin = (N_r, N_g, N_b))


filename = f"results/correlations/{lattice}_energy_{L1}x{L2}_{boundary_conditions_x}x{boundary_conditions_y}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}"
ma=HiddenFermion(n_elecs,"FFNN",n_hid,L1,L2,layers=layers,features=features,double_occupancy_bool=double_occupancy,MFinit=MFinitialization, hilbert=hi)

# ---------- define sampler ------------------------
if double_occupancy:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=ExchangeRule(graph=graph))
else:
  sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=tJExchangeRule(graph=graph))


vstate = nk.vqs.MCState(sa, ma, n_discard_per_chain=32, n_samples=n_samples, chunk_size=cs) 
vars = nkx.vqs.variables_from_file(filename+".mpack", vstate.variables)
# update the variables of vstate with the loaded data.
vstate.variables = vars
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# thermalize samples
for i in range(5):
 vstate.sample()

# -------------- calculate expectation values ---------------
flavors = [2,0,-2]
def lam3(hi, i):
    return (nc(hi, i, -2)-nc(hi,i,0))

def lam8(hi,i):
    return 1/(np.sqrt(3))*(nc(hi,i,-2)+nc(hi,i,0)-2*nc(hi,i,2))

def nh(hi,i):
    return 1-nc(hi,i,-2)-nc(hi,i,0)-nc(hi,i,2)


#spin-spin correlations
l33 = []
l88 = []
for dx in range(-L1//2, (L1+1)//2):
  for dy in range(-L2//2, (L2+1)//2):
    l33d = []
    l88d = []
    for i in graph.nodes():
        xi = i//L2
        yi = i%L2
        xj = (xi+dx)%L1
        yj = (yi+dy)%L2
        j = xj*L2+yj
        l33d.append(vstate.expect(lam3(hi,i)*lam3(hi,j)).mean/(np.sqrt(vstate.expect(lam3(hi,i)).variance)*np.sqrt(vstate.expect(lam3(hi,j)).variance)))
        l88d.append(vstate.expect(lam8(hi,i)*lam8(hi,j)).mean/(np.sqrt(vstate.expect(lam8(hi,i)).variance)*np.sqrt(vstate.expect(lam3(hi,j)).variance)))
    l33.append(np.mean(l33d))
    l88.append(np.mean(l88d))
print(l33)
print(l88)

np.save(f"results/correlations/{lattice}_l3l3_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}.npy", np.asarray(l33))
np.save(f"results/correlations/{lattice}_l8l8_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}.npy", np.asarray(l88))

#hole-spin-spin correlations
ds = np.asarray([[[1,0],[1,1],[2,-1]],[[0,1],[1,1],[-1,2]],[[-1,1],[-1,2],[-2,1]],[[-1,0],[-2,1],[-1,-1]],[[0,-1],[-1,-1],[1,-2]],[[1,-1],[1,-2],[2,-1]]])

C11 = []
for nn in range(ds.shape[0]):
  dx1 = ds[nn,0,0]
  dy1 = ds[nn,0,1]
  c_nn = []
  for dist in range(1,3):
    dx2 = ds[nn,dist,0]
    dy2 = ds[nn,dist,1]
    for i in graph.nodes():
      xi = i//L2
      yi = i%L2
      xj = (xi+dx1)%L1
      yj = (yi+dy1)%L2
      xl = (xi+dx2)%L1
      yl = (yi+dy2)%L2
      j = xj*L2+yj
      l = xl*L2+yl
      Sj = vstate.expect(lam3(hi,j))
      Sl = vstate.expect(lam3(hi,l))
      ni = vstate.expect(nh(hi,i))
      SjSlni = vstate.expect(lam3(hi,j)*lam3(hi,l)*nh(hi,i))
      c_nn.append((SjSlni.mean/(np.sqrt(Sj.variance*Sl.variance)*ni.mean)))
  C11.append(np.mean(c_nn))

np.save(f"results/new_symmetry/newJ/correlations/{lattice}_C3_h_11_av_new_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}.npy", np.asarray(C11))


ds = np.asarray([[[1,0],[-1,1]],[[0,1],[-1,0]],[[-1,1],[0,-1]],[[-1,0],[1,-1]],[[0,-1],[1,0]],[[1,-1],[0,1]]])

C11_sl = []
for nn in range(ds.shape[0]):
  dx1 = ds[nn,0,0]
  dy1 = ds[nn,0,1]
  dx2 = ds[nn,1,0]
  dy2 = ds[nn,1,1]
  c_nn = []
  o_sites, Si, Sj = [],[],[]
  for i in graph.nodes():
    xi = i//L2
    yi = i%L2
    xj = (xi+dx1)%L1
    yj = (yi+dy1)%L2
    xl = (xi+dx2)%L1
    yl = (yi+dy2)%L2
    j = xj*L2+yj
    l = xl*L2+yl 
    Sj = vstate.expect(lam3(hi,j))
    Sl = vstate.expect(lam3(hi,l))
    ni = vstate.expect(nh(hi,i))
    SjSlni = vstate.expect(lam3(hi,j)*lam3(hi,l)*nh(hi,i))
    c_nn.append((SjSlni.mean/(np.sqrt(Sj.variance*Sl.variance)*ni.mean)))
  C11_sl.append(np.mean(c_nn))

np.save(f"results/new_symmetry/newJ/correlations/{lattice}_C3_h_11_sl_av_new_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}.npy", np.asarray(C11_sl))

ds = np.asarray([[[1,0],[1,-1]],[[0,1],[1,0]],[[-1,1],[0,1]],[[-1,0],[-1,1]],[[0,-1],[-1,0]],[[1,-1],[0,-1]]])

C11_nn = []
for nn in range(ds.shape[0]):
  dx1 = ds[nn,0,0]
  dy1 = ds[nn,0,1]
  dx2 = ds[nn,1,0]
  dy2 = ds[nn,1,1]

  c_nn = []
  o_sites, Si, Sj = [],[],[]
  for i in graph.nodes():
    xi = i//L2
    yi = i%L2
    xj = (xi+dx1)%L1
    yj = (yi+dy1)%L2
    xl = (xi+dx2)%L1
    yl = (yi+dy2)%L2
    j = xj*L2+yj
    l = xl*L2+yl 
    Sj = vstate.expect(lam3(hi,j))
    Sl = vstate.expect(lam3(hi,l))
    ni = vstate.expect(nh(hi,i))
    SjSlni = vstate.expect(lam3(hi,j)*lam3(hi,l)*nh(hi,i))
    c_nn.append((SjSlni.mean/(np.sqrt(Sj.variance*Sl.variance)*ni.mean)))
  C11_nn.append(np.mean(c_nn))

np.save(f"results/new_symmetry/newJ/correlations/{lattice}_C3_h_11_nn_av_new_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nr={N_r}_Ng={N_g}_Nb={N_b}_t={t}_J={J}_mu={mu}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_MFinit={MFinitialization}.npy", np.asarray(C11_nn))


