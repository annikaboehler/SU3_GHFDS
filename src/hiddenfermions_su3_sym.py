from jax import numpy as jnp
import netket as nk
from jax.scipy.special import logsumexp
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
  symmetry: str="no symmetry"
  dtype: type = jnp.complex128
  need_b: bool=False
  number_b: int=0
  graph: nk.graph.Graph=None
  m3: int=0

  def setup(self):
    self.n_modes = 3*self.Lx*self.Ly
    self.orbitals = Orbitals(hilbert=self.hilbert, n_elecs=self.n_elecs,n_hid=self.n_hid,Lx=self.Lx, Ly=self.Ly, graph=self.graph,dtype=self.dtype, MFinit=self.MFinit)
    self.hidden = [nn.Dense(features=self.features,use_bias=True,param_dtype=self.dtype) for i in range(self.layers)]
    self.output = nn.Dense(features=self.n_hid*(self.n_elecs+self.n_hid),use_bias=True,param_dtype=self.dtype)
    self.a = self.param('a', zeros, (1,), dtype=self.dtype)
    if self.need_b:
      self.b = self.param('b', zeros, (self.number_b,), self.dtype) #needed if we couple GPUs

    # Pre-compute static lattice rotation arrays
    if self.symmetry == "lattice rotation" and self.graph is not None:
        self.gather_idx_120 = self._compute_gather_indices(graph=self.graph, theta=2 * np.pi / 3)
        self.gather_idx_240 = self._compute_gather_indices(graph=self.graph, theta=4 * np.pi / 3)



  def double_occupancy(self,x):
    x = x[:,:x.shape[-1]//3] + x[:,x.shape[-1]//3:2*x.shape[-1]//3] + x[:,2*x.shape[-1]//3:]
    return jnp.where(jnp.any(x > 1.5,axis=-1),True,False)

  def gen_spin_rotated_samples(self,x):
    x1 = x[:,:x.shape[1]//3].copy()
    x2 = x[:,(x.shape[1]//3):(2*x.shape[1]//3)].copy()
    x3 = x[:,(2*x.shape[1]//3):].copy()
    spin_perms = [[x1,x2,x3], [x2,x3,x1], [x3,x1,x2],
                 [x1,x3,x2], [x3,x2,x1], [x2,x1,x3]]
    #Signs for each permutation
    if (self.n_elecs//3)%2==1:
      perm_signs = [1,1,1,-1,-1,-1]
    else:
      perm_signs = [1,1,1,1,1,1]
    return spin_perms, jnp.asarray(perm_signs)

  def get_parity(self, array: jax.Array) -> jax.Array:
    """
    Count the parity of an array.
    This is the number of inversions in the array modulo 2.
    An inversion is a pair (i, j) such that i < j
    and array[i] > array[j].
    """
    if array.ndim == 1:
        array = array[jnp.newaxis, :]
    batch_dims = array.shape[:-1]
    inversion_matrix = array[..., :, jnp.newaxis] > array[..., jnp.newaxis, :]
    upper_triangular_mask = jnp.triu(
        jnp.ones((*batch_dims, array.shape[-1], array.shape[-1]), dtype=bool), k=1
    )
    inversion_count = jnp.sum(inversion_matrix & upper_triangular_mask, axis=(-2, -1))
    return 1-2*(inversion_count % 2)

  def _compute_gather_indices(self, graph, theta):
    # 1. Setup Dimensions
    n_sites = graph.n_nodes
    L = [self.Lx, self.Ly]
    # 2. Build the Index Map
    # We create an array where indices[new_site] = old_site
    # This allows us to just shuffle the columns of the batch.
    gather_indices = np.zeros(n_sites, dtype=np.int32)
    
    rot_120 = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])

    def pos_to_coord_general(pos, L):
        x_c, y_c = pos
        n2 = (2/np.sqrt(3)) * y_c
        n1 = x_c - 0.5 * n2
        n1 = int(np.round(np.mod(np.round(n1), L[0])))
        n2 = int(np.round(np.mod(np.round(n2), L[1])))
        return np.array([n1, n2])

    # Calculate mapping for every site (0 to N)
    for src_site in range(n_sites):
        pos = graph.positions[src_site]
        rotated_pos = rot_120 @ pos
        
        # Calculate where this site moves to
        new_coords = pos_to_coord_general(rotated_pos, L)
        dst_site = graph.id_from_basis_coords(np.array([new_coords[0], new_coords[1], 0]))
        
        # Record the mapping: The destination index pulls data from the source index
        gather_indices[dst_site] = src_site
 
    return jnp.array(gather_indices)
  
  def gen_lattice_rotated_samples(self,x, gather_indices):
    n_sites = self.graph.n_nodes
    batch = x.shape[0]
    gather_indices_full = jnp.repeat(jnp.expand_dims(gather_indices, axis=0), x.shape[0], axis=0)
    
    # Slice the channels
    red = x[:, :n_sites]
    green = x[:, n_sites:2*n_sites]
    blue = x[:, 2*n_sites:]

    red_occ = red.nonzero(size=(self.n_elecs+2)//3*batch)
    green_occ = green.nonzero(size=(self.n_elecs+1)//3*batch)
    blue_occ = blue.nonzero(size=(self.n_elecs)//3*batch)

    red_rot = gather_indices_full[red_occ[0], red_occ[1]]
    green_rot = gather_indices_full[green_occ[0], green_occ[1]]
    blue_rot = gather_indices_full[blue_occ[0], blue_occ[1]]

    red_rot = red_rot.reshape(batch, (self.n_elecs+2)//3)
    green_rot = green_rot.reshape(batch, (self.n_elecs+1)//3)
    blue_rot = blue_rot.reshape(batch, (self.n_elecs)//3)
   
    sign_red = self.get_parity(red_rot)
    sign_green = self.get_parity(green_rot)
    sign_blue = self.get_parity(blue_rot)

    total_sign = sign_red * sign_green * sign_blue

    # Apply the permutation to the batch axis (axis 0 is preserved automatically)
    new_indices = jnp.argsort(gather_indices)
    red_rot = red[:, new_indices]
    green_rot = green[:, new_indices]
    blue_rot = blue[:, new_indices]
    new_sample = jnp.concatenate([red_rot, green_rot, blue_rot], axis=1)
    
    return new_sample, total_sign

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
      return x, jnp.log(sign + 0j)
    else:
      return x - 1e12*do, jnp.log(sign + 0j)

  @nn.compact
  def __call__(self,x):
    do = self.double_occupancy(x)
    batch = x.shape[0]
   
    if self.symmetry=="spin rotation":
        spin_perms, perm_signs = self.gen_spin_rotated_samples(x)
        for j,perm in enumerate(spin_perms):
            if j!=0:
                xr_ = jnp.concatenate([perm[i] for i in range(len(perm))], axis=1)
                xr = jnp.concatenate([xr, xr_], axis=0)
                #print(self.symmetry)
    
        logpsi, logsign = self.calc_psi(xr)
        psi       = jnp.exp(logpsi)
        sign      = jnp.exp(logsign)
        psi0      = psi[0:batch] 
        sign0     = sign[0:batch]
        permsign0 = perm_signs[0]*jnp.ones((batch,))
        psi1      = psi[batch:2*batch]
        sign1     = sign[batch:2*batch]
        permsign1 = perm_signs[1]*jnp.ones((batch,))
        psi2      = psi[2*batch:3*batch]
        sign2     = sign[2*batch:3*batch]
        permsign2 = perm_signs[2]*jnp.ones((batch,))
        psi3      = psi[3*batch:4*batch]
        sign3     = sign[3*batch:4*batch]
        permsign3 = perm_signs[3]*jnp.ones((batch,))
        psi4      = psi[4*batch:5*batch]
        sign4     = sign[4*batch:5*batch]
        permsign4 = perm_signs[4]*jnp.ones((batch,))
        psi5      = psi[5*batch:]
        sign5     = sign[5*batch:]
        permsign5 = perm_signs[5]*jnp.ones((batch,))
        log_psi = jnp.log(1/6*(permsign0*psi0+permsign1*psi1+permsign2*psi2+permsign3*psi3+permsign4*psi4+permsign5*psi5))+jnp.log(sign[0:batch])
    
    elif self.symmetry=="lattice rotation":
        print(self.symmetry)
        rot_samples_120, signs120 = self.gen_lattice_rotated_samples(x, self.gather_idx_120)
        rot_samples_240, signs240 = self.gen_lattice_rotated_samples(x, self.gather_idx_240)
        xr = jnp.concatenate([x, rot_samples_120, rot_samples_240], axis=0)
        
        perm_signs = jnp.concatenate([jnp.ones(batch), signs120, signs240], axis=0)
        logpsi, logsign = self.calc_psi(xr)
        log_psi_total = logsign + logpsi
        log_psi_total = log_psi_total.reshape(3, batch)
        phases = jnp.array([0, 2*jnp.pi/3*self.m3, 4*jnp.pi/3*self.m3])
        weights = perm_signs.reshape(3, batch) * jnp.exp(1j * phases[:, None])
        log_psi = logsumexp(log_psi_total, axis=0, b=weights) - jnp.log(3)

    else:
        log_psi, logsign = self.calc_psi(x)
        log_psi = log_psi + logsign
    
    if self.double_occupancy_bool:
      return log_psi
    else:
      return log_psi - 1e12*do #need this to make sure connected elements that lead to do have prob 0


class Orbitals(nn.Module):

  hilbert: HomogeneousHilbert
  n_elecs: int
  n_hid: int
  Lx: int
  Ly: int
  graph: nk.graph
  MFinit: str
  dtype: type
  stop_grad_mf: bool=False

  def _init_orbitals_dct(self, key, shape):
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
        elif self.dtype==jnp.complex128:
          res = jnp.exp(1j*2*jnp.pi*(kx/self.Lx*x + ky/self.Ly*y))
        return res

    def ft(k_arr, max_val):
        matrix = []
        for idx,(kx, ky) in enumerate(k_arr[:max_val]):
          kstate = [ft_local_pbc(x,y,kx,ky) for y in range(self.Ly) for x in range(self.Lx)]
          matrix.append(kstate)
        return jnp.array(matrix, dtype=self.dtype)

    n_elecs = shape[1]
    k_modes = []
    for kx in range(0, self.Lx):
      for ky in range(0, self.Ly):
        k_modes.append((kx,ky))
    sorted_k_modes = sorted(k_modes, key=lambda x: (-np.cos(2*np.pi*x[0]/self.Lx) - np.cos(2*np.pi*x[1]/self.Ly)-np.cos(2*np.pi*(x[0]/self.Lx-x[1]/self.Ly)), x))
    k_arr = np.array(sorted_k_modes)

    rmatrix = ft(k_arr, self.hilbert.n_fermions_per_spin[0])
    gmatrix = ft(k_arr, self.hilbert.n_fermions_per_spin[1])
    bmatrix = ft(k_arr, self.hilbert.n_fermions_per_spin[2])
    mf = jnp.block([[rmatrix, jnp.zeros(rmatrix.shape, dtype=self.dtype), jnp.zeros(rmatrix.shape, dtype=self.dtype)], [jnp.zeros(gmatrix.shape, dtype=self.dtype), gmatrix, jnp.zeros(gmatrix.shape, dtype=self.dtype)],
                        [jnp.zeros(bmatrix.shape, dtype=self.dtype), jnp.zeros(bmatrix.shape, dtype=self.dtype), bmatrix]]).T
    jax.debug.print("mf={x}",x=mf)
    return mf
  
  def _init_orbitals_ED(self, key, shape):
    def Hk(t):
      N = self.graph.n_nodes
      H = np.zeros([N,N])
      for (u,v) in self.graph.edges():
        H[u,v] += -t
      H += H.conj().T

      en, us = np.linalg.eigh(H)
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
                        [jnp.zeros(bmatrix.shape), jnp.zeros(bmatrix.shape), bmatrix]]).T
    jax.debug.print("mf={x}",x=mf)
    return mf

  def _init_orbitals_obc(self, key, shape):
    def Hk(t):
      # define single particle Hamiltonian: literally have hopping from site to site
      N = self.Lx*self.Ly # number of sites
      H = np.zeros([N,N], dtype=self.dtype)
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
      en, us = np.linalg.eigh(H)
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
          mat[i, rcnt] = psi
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
    if self.MFinit=="Fermi_ED":
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_ED,(3*self.Lx*self.Ly,self.n_elecs))
    elif self.MFinit=="Fermi": #still square lattice, use Fermi_ED for traingular
        orbitals_mfmf = self.param('orbitals_mf', self._init_orbitals_dct, (3*self.Lx*self.Ly, self.n_elecs))
    elif self.MFinit=="Fermi_obc": #still square lattice, use Fermi_ED for traingular
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_obc,(3*self.Lx*self.Ly,self.n_elecs))
    elif self.MFinit=="random":
        orbitals_mfmf = self.param('orbitals_mf', normal(0.5, dtype=self.dtype),(3*self.Lx*self.Ly,self.n_elecs))
    else:
        raise NotImplementedError("This MF initialization is not implemented! Chose one of: Fermi_ED, Fermi, random, Fermi_obc")
    orbitals_mfhf = self.param('orbitals_hf', zeros,(3*self.Lx*self.Ly,self.n_hid), dtype=self.dtype)
    if self.stop_grad_mf: 
        orbitals_mfmf = jax.lax.stop_gradient(orbitals_mfmf)
    orbitals = jnp.concatenate((orbitals_mfmf, orbitals_mfhf), axis=1, dtype=self.dtype)
    ind1, ind2 = jnp.nonzero(x,size=x.shape[0]*self.n_elecs)
    x = jnp.repeat(jnp.expand_dims(orbitals,0),x.shape[0],axis=0)[ind1,ind2]
    return x.reshape(-1,self.n_elecs,x.shape[1])

