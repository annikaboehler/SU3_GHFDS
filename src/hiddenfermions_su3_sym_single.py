from jax import numpy as jnp
import netket as nk
import jax
from jax.random import PRNGKey, choice, split
from functools import partial
from flax import linen as nn
from jax.nn.initializers import zeros, normal, constant
from netket.utils.dispatch import dispatch
from netket import experimental as nkx
from netket.jax import apply_chunked
import numpy as np
from netket.hilbert.homogeneous import HomogeneousHilbert
from itertools import permutations
from jax.scipy.special import logsumexp

class HiddenFermion(nn.Module):
  n_elecs: int
  network: str
  n_hid: int
  Lx: int
  Ly: int
  hilbert: HomogeneousHilbert
  layers: int=2
  features: int=4
  double_occupancy_bool: bool=False
  MFinit: str="Fermi"
  symmetry: str="spin rotation"
  dtype: type = jnp.float64
  need_b: bool=False

  def setup(self):
    self.n_modes = 3*self.Lx*self.Ly
    self.orbitals = Orbitals(self.hilbert, self.n_elecs,self.n_hid,self.Lx, self.Ly, self.MFinit, dtype=self.dtype)
    self.hidden = [nn.Dense(features=self.features,use_bias=True,param_dtype=self.dtype) for i in range(self.layers)]
    self.output = nn.Dense(features=self.n_hid*(self.n_elecs+self.n_hid),use_bias=True,param_dtype=self.dtype)
    self.a = self.param('a', zeros, (1,), dtype=self.dtype)
    if self.need_b:
      self.b = self.param('b', zeros, (3,), self.dtype) #needed if we couple 2GPUs?

  def double_occupancy(self,x):
    x = x[:,:x.shape[-1]//3] + x[:,x.shape[-1]//3:2*x.shape[-1]//3] + x[:,2*x.shape[-1]//3:]

    return jnp.where(jnp.any(x > 1.5,axis=-1),True,False)

  def gen_spin_rotated_samples(self,x):
    x1 = x[:,:x.shape[1]//3].copy()
    x2 = x[:,(x.shape[1]//3):(2*x.shape[1]//3)].copy()
    x3 = x[:,(2*x.shape[1]//3):].copy()
    spin_configs = [x1,x2,x3]
    spin_perms = permutations(spin_configs)
    #jax.debug.print("spin_perms={x}", x=spin_perms)
    # Concatenate arrays in each permutation order
    #x_ = [np.concatenate(perm, axis=1) for perm in spin_perms]
    return spin_perms

  def selu(self,x):
    if self.dtype==jnp.float64:
      return jax.nn.selu(x)
    else:
      return jax.nn.selu(x.real) +1j*jax.nn.selu(x.imag)

  def calc_psi(self,x,return_orbs=False):
    orbitals = self.orbitals(x)
    do = self.double_occupancy(x)
    for i in range(self.layers):
        x = self.selu(self.hidden[i](x))
    x_ = self.output(x).reshape(x.shape[0],self.n_hid,self.n_elecs + self.n_hid)

    x_ = self.a*x_
    x_ += jnp.concatenate((jnp.zeros((x.shape[0], self.n_hid, self.n_elecs)), jnp.repeat(jnp.expand_dims(jnp.eye(self.n_hid), axis=0),x.shape[0],axis=0)), axis=2)
    x = jnp.concatenate((orbitals,x_),axis=1)
    sign, x = jnp.linalg.slogdet(x)
    if self.double_occupancy_bool:
      return x, sign
    else:
      return x - 1e12*do, sign

  @nn.compact
  def __call__(self,x):
    do = self.double_occupancy(x)
    batch = x.shape[0]
    xr = x.copy()
    #jax.debug.print("xshape={x}", x=xr.shape)
    logpsi, sign = self.calc_psi(x)
    psi = jnp.exp(sign+logpsi)
    if self.symmetry=="spin rotation":
      spin_perms = self.gen_spin_rotated_samples(x)
      #jax.debug.print("spin perms={x}", x=spin_perms)
      for j,perm in enumerate(spin_perms):
        if j!=0:
          #jax.debug.print("perm={x}", x=[perm[i].shape for i in range(len(perm))])
          xr_ = jnp.concatenate([perm[i] for i in range(len(perm))], axis=1)
          #jax.debug.print("perm shape={x}", x=xr_.shape)
          xr = jnp.concatenate([xr, xr_], axis=0)
          #jax.debug.print("xr={x}", x=xr.shape)
      #jax.debug.print("final shape={x}", x=xr.shape)
      print(self.symmetry)
    else:
        raise NotImplementedError("This symmetry is not implemented!")
    logpsi, sign = self.calc_psi(xr)
    logpsi_rs = jnp.reshape(logpsi, (6,-1)).T
    sign_rs = jnp.reshape(sign, (6,-1)).T
    #psi       = jnp.exp(logpsi)
    #sign      = jnp.exp(logsign)
    #psi0      = psi[0:batch] 
    #sign0     = sign[0:batch]
    #psi1      = psi[batch:2*batch]
    #sign1     = sign[batch:2*batch]
    #psi2      = psi[2*batch:3*batch]
    #sign2     = sign[2*batch:3*batch]
    #psi3      = psi[3*batch:4*batch]
    #sign3     = sign[3*batch:4*batch]
    #psi4      = psi[4*batch:5*batch]
    #sign4     = sign[4*batch:5*batch]
    #psi5      = psi[5*batch:]
    #sign5     = sign[5*batch:]
    #jax.debug.print("psi shape={psi}", psi=(psi0.shape, psi1.shape, psi5.shape))
    #log_psi = jnp.log(1/6*(psi0*sign0+psi1*sign1+psi2*sign2+psi3*sign3+psi4*sign4+psi5*sign5))
    log_psi = logsumexp(a=logpsi_rs, b=sign_rs, axis=1)
    print("logpsi shape", log_psi.shape)
    #jax.debug.print("log_psi={x}", x=log_psi)
    if self.double_occupancy_bool:
      return log_psi
    else:
      return log_psi - 1e12*do


class Orbitals(nn.Module):

  hilbert: HomogeneousHilbert
  n_elecs: int
  n_hid: int
  Lx: int
  Ly: int
  MFinit: str
  dtype: type=jnp.float64
  stop_grad_mf: bool=False

  def _init_orbitals_dct(self, key, shape, dtype):
    def ft_local_pbc(x,y,kx,ky):
        if self.dtype==jnp.float64:
          if kx<=self.Lx//2 and ky<=self.Ly//2:
              res = jnp.cos(2*jnp.pi*(x)/self.Lx*(kx))*jnp.cos(2*jnp.pi*(y)/self.Ly*(ky))
          elif kx>=self.Lx//2 and ky<=self.Ly//2:
              res = jnp.sin(2*jnp.pi*(x)/self.Lx*(kx))*jnp.cos(2*jnp.pi*(y)/self.Ly*(ky)) 
          elif kx<=self.Lx//2 and ky>=self.Ly//2:
              res = jnp.cos(2*jnp.pi*(x)/self.Lx*(kx))*jnp.sin(2*jnp.pi*(y)/self.Ly*(ky)) 
          elif kx>=self.Lx//2 and ky>=self.Ly//2:
              res = jnp.sin(2*jnp.pi*(x)/self.Lx*(kx))*jnp.sin(2*jnp.pi*(y)/self.Ly*(ky)) 
        else:
          res = jnp.exp(1j*2*jnp.pi*(kx/self.Lx*x + ky/self.Ly*y))
        return res

    def ft(k_arr, max_val):
        matrix = []
        for idx,(kx, ky) in enumerate(k_arr[:max_val]):
          kstate = [ft_local_pbc(x,y,kx,ky) for y in range(self.Ly) for x in range(self.Lx)]
          matrix.append(kstate)
        return jnp.array(matrix)

    n_elecs = shape[1]
    k_modes = []
    for kx in range(0, self.Lx):
      for ky in range(0, self.Ly):
        k_modes.append((kx,ky))
    sorted_k_modes = sorted(k_modes, key=lambda x: (-np.cos(2*np.pi*x[0]/self.Lx) - np.cos(2*np.pi*x[1]/self.Ly), x))
    k_arr = np.array(sorted_k_modes)

    rmatrix = ft(k_arr, self.hilbert.n_fermions_per_spin[0])
    gmatrix = ft(k_arr, self.hilbert.n_fermions_per_spin[1])
    bmatrix = ft(k_arr, self.hilbert.n_fermions_per_spin[2])
    mf = jnp.block([[rmatrix, jnp.zeros(rmatrix.shape, dtype=self.dtype), jnp.zeros(rmatrix.shape, dtype=self.dtype)], [jnp.zeros(gmatrix.shape, dtype=self.dtype), gmatrix, jnp.zeros(gmatrix.shape, dtype=self.dtype)],
                        [jnp.zeros(bmatrix.shape, dtype=self.dtype), jnp.zeros(bmatrix.shape, dtype=self.dtype), bmatrix]]).T
    jax.debug.print("mf={x}",x=mf)
    return mf


  def _init_orbitals_obc(self, key, shape, dtype):
    def Hk(t):
      # define single particle Hamiltonian: literally have hopping from site to site
      N = self.Lx*self.Ly # number of sites
      H = np.zeros([N,N])
      for x in range(self.Lx):
        for y in range(self.Ly):
          i = x*self.Lx + y # map 1D to 2D
          # hopping
          if x<self.Lx-1:
            ix = (x+1)*self.Ly+y
            H[i,ix] += -t
            H[ix,i] += -t
          if y<self.Ly-1:
            iy = x*self.Ly + (y+1)
            H[i,iy] += -t
            H[iy,i] += -t
      en, us = np.linalg.eig(H)
      return en, us


    def initialize_obc(num_particles):
      # 'orbitals' are now just eigenstates of single particle Hamiltonian (run from 0 to mX*mY-1) 
      ks = range(self.Lx*self.Ly)
      # find possible r-states
      rs = [[x,y] for y in range(self.Ly) for x in range(self.Lx)]

      mat = np.zeros([num_particles,len(rs)], dtype=self.dtype)
      # get single particle eigenenergies and states
      en, us = Hk(1)
      ## sort energies
      indices = sorted(range(len(en)),key=lambda index: en[index])
      energies=en[indices]
      us = us[:,indices]
      for i in range(num_particles):
        rcnt=0
        for r in rs:
          psi = us[rcnt,i] # wave function coefficient for this eigenstate + position r
          mat[i, rcnt] = np.real(psi) 
          rcnt+=1
      
      return mat

    n_elecs = shape[1]
    k_modes = []
    for kx in range(0, self.Lx):
      for ky in range(0, self.Ly):
        k_modes.append((kx,ky))
    sorted_k_modes = sorted(k_modes, key=lambda x: (-np.cos(2*np.pi*x[0]/self.Lx) - np.cos(2*np.pi*x[1]/self.Ly), x))
    k_arr = np.array(sorted_k_modes)

    rmatrix = initialize_obc(self.hilbert.n_fermions_per_spin[0])
    gmatrix = initialize_obc(self.hilbert.n_fermions_per_spin[1])
    bmatrix = initialize_obc(self.hilbert.n_fermions_per_spin[2])
    mf = jnp.block([[rmatrix, jnp.zeros(rmatrix.shape, dtype=self.dtype), jnp.zeros(rmatrix.shape, dtype=self.dtype)], [jnp.zeros(gmatrix.shape, dtype=self.dtype), gmatrix, jnp.zeros(gmatrix.shape, dtype=self.dtype)],
                        [jnp.zeros(bmatrix.shape, dtype=self.dtype), jnp.zeros(bmatrix.shape, dtype=self.dtype), bmatrix]]).T
    jax.debug.print("mf={x}",x=mf)
    return mf



  @nn.compact
  def __call__(self,x):
    if self.MFinit=="Fermi":
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_dct,(3*self.Lx*self.Ly,self.n_elecs), dtype=self.dtype)
    elif self.MFinit=="Fermi_obc":
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_obc,(3*self.Lx*self.Ly,self.n_elecs))
    elif self.MFinit=="random":
        orbitals_mfmf = self.param('orbitals_mf', normal(0.1),(3*self.Lx*self.Ly,self.n_elecs))
    else:
        raise NotImplementedError("This MF initialization is not implemented! Chose one of: Fermi, random, Fermi_obc, HF")
    orbitals_mfhf = self.param('orbitals_hf', zeros,(3*self.Lx*self.Ly,self.n_hid), dtype=self.dtype)
    if self.stop_grad_mf: 
        orbitals_mfmf = jax.lax.stop_gradient(orbitals_mfmf)
    orbitals = jnp.concatenate((orbitals_mfmf, orbitals_mfhf), axis=1)
    ind1, ind2 = jnp.nonzero(x,size=x.shape[0]*self.n_elecs)
    #jax.debug.print("x={x}", x=x)
    #jax.debug.print("ind={i}", i=(ind1, ind2))
    #jax.debug.print("orbs={o}", o=orbitals)
    x = jnp.repeat(jnp.expand_dims(orbitals,0),x.shape[0],axis=0)[ind1,ind2]
    #jax.debug.print("returns={x}", x=x.reshape(-1,self.n_elecs,x.shape[1]))
    return x.reshape(-1,self.n_elecs,x.shape[1])
