from md.mdunits import MDUnits
import torch
from ase import Atoms

class System:

  def __init__(self, device="cuda", initializer=None):
    self.device = device

    self.atom_numbers = None
    self.masses = None
    self.positions = None
    self.momenta = None
    self.forces = None
    self.energy = None 

    self.cells = None
    self.pbc = None
    self.stress = None

    self.properties = {}

    self.initializer = initializer

    self.step = 0

  def load_ase_object(self, atoms):
    self.n_atoms = len(atoms.numbers)
    self.atom_numbers = torch.zeros(self.n_atoms)
    self.masses = torch.ones(self.n_atoms)
    self.positions = torch.zeros(self.n_atoms, 3, device=self.device)
    self.momenta = torch.zeros(self.n_atoms, 3, device=self.device)
    self.cells = torch.zeros(3, 3, device=self.device)
    self.pbc = torch.zeros(3, device=self.device)

    self.atom_numbers = torch.from_numpy(atoms.get_atomic_numbers())
    self.masses = torch.from_numpy(atoms.get_masses() * MDUnits.da2internal)
    self.positions = torch.from_numpy(atoms.get_positions() * MDUnits.angs2internal)
    self.cells = torch.from_numpy(atoms.cell * MDUnits.angs2internal)
    self.pbc = torch.from_numpy(atoms.pbc).bool()

    # init momenta
    if self.initializer is None:
      raise ValueError("Initializer must be specified")

    self.initializer.initialize_system(self)

  @property
  def ase_object(self):
    atom_numbers = self.atom_numbers.cpu().detach().numpy()
    positions = self.positions.cpu().detach().numpy() / MDUnits.angs2internal
    cells = self.cells.cpu().detach().numpy() / MDUnits.angs2internal
    pbc = self.pbc.cpu().detach().numpy()

    system = Atoms(atom_numbers, positions, cell=cells, pbc=pbc)
    return system

  @property
  def kinetic_energy(self):
    kin_energy = 0.5 * self.momenta ** 2 / self.masses[:, None]
    return kin_energy

  @property
  def temperature(self):
    temperature = torch.sum(
      2.0 / (3.0 * MDUnits.kB) * self.kinetic_energy, 1
    )

    return temperature

  def remove_com(self):
    self.positions = self.positions


class Simulator:

  def __init__(
      self, system, integrator, calculator, simulator_hooks=[], step=0, restart=False
    ):
      self.system = system
      self.integrator = integrator
      self.calculator = calculator
      self.simulator_hooks = simulator_hooks
      self.step = step
      self.n_steps = None
      self.restart = restart

  def simulate(self, n_steps=1000):

    self.n_steps = n_steps

    # calculate forces
    if self.system.forces == None:
      self.calculator.calculate(self.system)

    # Call hooks at the simulation start
    for hook in self.simulator_hooks:
      hook.on_simulation_start(self)

    for n in range(n_steps):
      # Call hook berfore first half step
      for hook in self.simulator_hooks:
          hook.on_step_begin(self)

      print("In step {}".format(n))

      # Do half step momenta
      self.integrator.half_step(self.system)

      # Do propagation MD/PIMD
      self.integrator.main_step(self.system)

      # Compute new forces
      self.calculator.calculate(self.system)

      # Call hook after forces
      for hook in self.simulator_hooks:
          hook.on_step_middle(self)

      # Do half step momenta
      self.integrator.half_step(self.system)

      # Call hooks after second half step
      # Hooks are called in reverse order to guarantee symmetry of
      # the propagator when using thermostats and barostats
      for hook in self.simulator_hooks[::-1]:
          hook.on_step_end(self)

      self.step += 1

    # Call hooks at the simulation end
    for hook in self.simulator_hooks:
      hook.on_simulation_end(self)
