from ase.calculators.emt import EMT
from ase.optimize.lbfgs import LBFGS

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic, BodyCenteredCubic

import random
import numpy as np
from numpy.core.arrayprint import get_printoptions

def get_magnitude(vec):
  return np.sqrt(vec[0] ** 2 + vec[1] **2 + vec[2]**2)

R = 8.314

class SSWTrajectory:

  def __init__(self,
  structure,
  l_lambda = 0.5,
  H=10,
  ds=0.5,
  w=0.5,
  T=300,
  f_max_initial=0.001,
  f_max_inter=1,
  f_max_final=1
  ):

    # properties
    self.structure = structure
    self.n_atoms = structure.get_global_number_of_atoms()
    self.T = T

    # settings
    self.l_lambda = l_lambda
    self.H = H
    self.ds = ds
    self.w = w
    self.f_max_initial = f_max_initial
    self.f_max_inter = f_max_inter
    self.f_max_final = f_max_final
    self.use_BPCBD = False

    # generate initial minima
    opt = LBFGS(self.structure)
    opt.run(fmax=self.f_max_initial)
    self.R0_h = self.structure    # R0_0

    # output array
    self.output = []

  def generate(self, n_steps):

    for i in range(n_steps):
      # global direction
      MaxwellBoltzmannDistribution(self.structure, temperature_K=self.T)
      N0_g = self.structure.get_momenta() / np.array([list(self.structure.get_masses()) * 3]).reshape(self.n_atoms, 3)

      # local direction
      pair = []
      while len(pair) == 0:
        idx1 = int(random.random() * self.n_atoms) - 1
        idx2 = int(random.random() * self.n_atoms) - 1

        if get_magnitude(self.structure.get_positions()[idx1] - self.structure.get_positions()[idx2]) > 3:
          pair = [idx1, idx2]

      swapped_positions = self.structure.get_positions().copy()
      swapped_positions[[idx1, idx2]] = swapped_positions[[idx2, idx1]]
      N0_l = swapped_positions - self.structure.get_positions()

      # initial direction
      N0_i = (N0_g.flatten() + self.l_lambda * N0_l.flatten()) / get_magnitude(N0_g.flatten() + self.l_lambda * N0_l.flatten())
      
      # Climbing procedure
      bias_forces = np.zeros(self.n_atoms * 3)
      Rn_h = self.R0_h

      tmp_output = []
      tmp_output.append(Rn_h.get_potential_energy())

      h = 0
      while h < self.H or Rn_h.get_potential_energy() < R0_h.get_potential_energy(): 

        # calculate Nn_i
        if self.use_BPCBD:
          Nn_i = N0_i
        else:
          Nn_i = N0_i

        # displace along Nn_i
        new_postions = Rn_h.get_positions().flatten() + Nn_i * self.ds
        Rn_h.set_positions(new_postions.reshape(self.n_atoms, 3))
        bias_forces += self.w * np.exp( - ((R0_h.get_positions().flatten() - Rn_h.get_positions().flatten()) * Nn_i)**2 / ( 2 * self.ds **2)) * (R0_h.get_positions().flatten() - Rn_h.get_positions().flatten()) * Nn_i / self.ds ** 2 * Nn_i

        # relax the structure on the modified PES
        opt = LBFGS(Rn_h, bias=bias_forces.reshape(self.n_atoms, 3))
        opt.run(fmax=self.f_max_inter)

        tmp_output.append(Rn_h.get_potential_energy())

        h += 1

      # optimizer the structure on the real PES
      opt = LBFGS(Rn_h)
      opt.run(fmax=self.f_max_final)

      tmp_output.append(Rn_h.get_potential_energy())

      self.output.append(tmp_output)

      # Metropolis Monte Carlo Decision
      if Rn_h.get_potential_energy() < R0_h.get_potential_energy():
        # accept next step
        R0_h = Rn_h
      else:
        P = np.exp((R0_h.get_potential_energy() - Rn_h.get_potential_energy()) / R * self.T)
        if random.random() < P:
          # accept next step
          R0_h = Rn_h

  @property
  def get_results(self):
    return self.output

# print('\n')
# print('REsults of the walk:')
# print(output)

# import matplotlib.pyplot as plt

# for arr in output:
#   plt.plot(np.arange(len(arr)), arr)

# plt.savefig("testplot.png")