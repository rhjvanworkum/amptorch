

class Integrator:

  def __init__(self, time_step, detach=False, device="cuda"):
    self.time_step = time_step # * conversion
    self.detach = detach
    self.device = device

  def main_step(self, system):
    self._main_step(system)
    if self.detach:
      system.positions = system.positions.detach()
      system.momenta = system.momenta.detach()

  def half_step(self, system):
    system.momenta = system.momenta + 0.5 * system.forces * self.time_step

  def _main_step(self, system):
    """ Implemented by derived routine"""
    raise NotImplementedError


class VerletVelocity(Integrator):

  def __init__(self, time_step, detach=True, device="cuda"):
    super(VerletVelocity, self).__init__(time_step, detach=detach, device=device)

  def _main_step(self, system):
    system.positions = system.positions + self.time_step * system.momenta / system.masses[:, None]