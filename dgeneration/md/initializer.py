import torch

class MaxwellBoltzmannInit:

  def __init__(self, temperature, remove_translation=False, remove_rotation=False):
    self.temperature = temperature
    self.remove_translation = remove_translation
    self.remove_rotation = remove_rotation

  def initialize_system(self, system):
    self.setup_momenta(system)

  def setup_momenta(self, system):

    # Move center of mass to origin
    system.remove_com()

    momenta = (
      torch.randn(system.momenta.shape, device=system.device)
      * system.masses[:, None]
    )

    system.momenta = momenta

    # Remove translational motion if requested
    if self.remove_translation:
        system.remove_com_translation()

    # Remove rotational motion if requested
    if self.remove_rotation:
        system.remove_com_rotation()

    scaling = torch.sqrt(self.temperature / system.temperature)
    system.momenta *= scaling[:, None]

    # if torch.sum(system.temperature) != (system.n_atoms * self.temperature):
    #   raise ValueError("Initializing temperature wrent wrong, expected {}, but got {} instead".format(self.temperature, system.temperature))
    print("Systems temperature initialized to {}".format(system.temperature))