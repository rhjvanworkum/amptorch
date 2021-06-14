import torch

class SimulationHook:

  def on_step_begin(self, simulator):
    pass

  def on_step_middle(self, simulator):
    pass

  def on_step_end(self, simulator):
    pass

  def on_simulation_start(self, simulator):
    pass

  def on_simulation_end(self, simulator):
    pass


class CollectiveVariable:
    """
    Basic collective variable to be used in combination with the :obj:`MetaDyn` hook.
    The ``_colvar_function`` needs to be implemented based on the desired collective
    variable.

    Args:
        width (float): Parameter regulating the standard deviation of the Gaussians.
    """

    def __init__(self, width):
        # Initialize the width of the Gaussian
        self.width = 0.5 * width ** 2

    def get_colvar(self, structure):
        """
        Compute the collecyive variable.

        Args:
            structure (torch.Tensor): Atoms positions taken from the system in the :obj:`schnetpack.md.Simulator`.

        Returns:
            torch.Tensor: Collective variable computed for the structure.
        """
        return self._colvar_function(structure)

    def _colvar_function(self, structure):
        """
        Placeholder for defining the particular collective variable function to be used.

        Args:
            structure (torch.Tensor): Atoms positions taken from the system in the :obj:`schnetpack.md.Simulator`.
        """
        raise NotImplementedError


class BondColvar(CollectiveVariable):
    """
    Collective variable acting on bonds between atoms.
    ``idx_a`` indicates the index of the first atom in the structure tensor, ``idx_b`` the second.
    Counting starts at zero.

    Args:
        idx_a (int): Index of the first atom in the positions tensor provided by the simulator system.
        idx_b (int): Index of the second atom.
        width (float): Width of the Gaussians applied to this collective variable. For bonds, units of Bohr are used.
    """

    def __init__(self, idx_a, idx_b, width):
        super(BondColvar, self).__init__(width)
        self.idx_a = idx_a
        self.idx_b = idx_b

    def _colvar_function(self, structure):
        """
        Compute the distance between both atoms.

        Args:
            structure (torch.Tensor): Atoms positions taken from the system in the :obj:`schnetpack.md.Simulator`.

        Returns:
            torch.Tensor: Bind collective variable.
        """
        vector_ab = structure[self.idx_b, :] - structure[self.idx_a, :]
        return torch.norm(vector_ab)


class MetaDynamics(SimulationHook):

  def __init__(self, collective_variables, frequency=200, weight=1.0 / 627.509, store_potential=True):
      self.collective_variables = collective_variables
      self.store_potential = store_potential
      
      self.gaussian_centers = None
      self.gaussian_mask = None
      self.collective_variable_widths = None

      self.frequency = frequency
      self.weigth = weight
      self.n_gaussians = 0

  def on_simulation_start(self, simulator):
      """
        Initialize the tensor holding the Gaussian centers and widths. These
        will be populated during the simulation.

        Args:
            simulator (schnetpack.md.Simulator): Main simulator used for driving the dynamics
      """
      n_gaussian_centers = int(simulator.n_steps / self.frequency) + 1
      self.gaussian_centers = torch.zeros(
          n_gaussian_centers,
          len(self.collective_variables),
          device=simulator.system.device
      )
      self.collective_variable_width = torch.FloatTensor(
        [cv.width for cv in self.collective_variables],
        device=simulator.system.device
      )
      self.gaussian_mask = torch.zeros(
        n_gaussian_centers, device=simulator.system.device
      )
      self.gaussian_mask[0] = 1

  def on_step_middle(self, simulator):
      """
      Based on the current structure, compute the collective variables and the
      associated Gaussian potentials. If multiple collective variables are given,
      a product potential is formed. torch.autograd is used to compute the forces
      resulting from the potential, which are then in turn used to update the system
      forces. A new Gaussian is added after a certain number of steps.

      Args:
          simulator (schnetpack.md.Simulator): Main simulator used for driving the dynamics
      """
      # Get and detach the structure from the simulator
      structure = simulator.system.positions.detach()
      # Enable gradients for bias forces
      structure.requires_grad = True

      # Compute the collective variables
      colvars = torch.stack(
          [colvar.get_colvar(structure) for colvar in self.collective_variables],
      )

      # Compute the Gaussians for the potential
      gaussians = torch.exp(
          - ((colvars - self.gaussian_centers) ** 2) / self.collective_variable_width
      )
      # Compute the bias potential and apply mask for centers not yet stored
      bias_potential = torch.prod(gaussians) * self.gaussian_mask

      # Finalize potential and compute forces
      bias_potential = torch.sum(self.weigth * bias_potential)
      bias_forces = -torch.autograd.grad(
          bias_potential, structure, torch.ones_like(bias_potential)
      )[0]

      # Store bias potential, collective variables and update system forces
      simulator.system.properties["bias_potential"] = bias_potential.detach()
      simulator.system.properties["collective_variables"] = colvars.detach()
      simulator.system.forces = simulator.system.forces + bias_forces.detach().numpy()

      if self.store_potential:
          # TODO: Much better to move this to a state dict?
          # Store information on the general shape of the bias potential
          simulator.system.properties[
              "gaussian_centers"
          ] = self.gaussian_centers.detach()
          simulator.system.properties[
              "gaussian_widths"
          ] = self.collective_variable_width.detach()
          simulator.system.properties["gaussian_mask"] = self.gaussian_mask.detach()

      # Add a new Gaussian to the potential every n_steps
      if simulator.step % self.frequency == 0:
          self.gaussian_centers[self.n_gaussians] = colvars.detach()
          # Update the mask
          self.gaussian_mask[self.n_gaussians + 1] = 1
          self.n_gaussians += 1
