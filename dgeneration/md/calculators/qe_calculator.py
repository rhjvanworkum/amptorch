from md.mdunits import MDUnits
from ase.calculators.espresso import Espresso
from ase.calculators.emt import EMT
from ase.db import connect

import torch
import numpy as np

class MDCalculator:

  def __init__(self,
  force_conversion=1.0,
  property_conversion={},
  stress_handle=None,
  stress_conversion="eV / Angstrom / Angstrom / Angstrom",
  detach=True,
  ):
    self.results = {}
    self.stress_handle = stress_handle

    self.position_conversion = MDUnits.angs2internal
    self.force_conversion = MDUnits.unit2internal(force_conversion)
    self.stress_handle = None
    self.stress_converison = None

    self.detach = detach

  def calculate(self):
    raise NotImplementedError

  # TODO:  check this
  def _update_system(self, system):
    for p in ["energy", "forces"]:
      if p not in self.results:
        raise ValueError("Request property {:s} not in results".format(p))
      else:
        if self.detach:
          self.results[p] = self.results[p].detach()

    system.forces = (
      self.results["forces"] * self.force_conversion
    )
        


class QECalculator(MDCalculator):

  def __init__(
    self,
    force_conversion=0.5,
    property_conversion={},
    stress_handle=None,
    stress_conversion="eV / Angstrom / Angstrom / Angstrom",
    detach=False,
    database_file=None,
    pseudopotentials=None,
    tstress=True,
    tprnfor=True,
    kpts=(1,1,1)
    ):
      super(QECalculator, self).__init__(
        force_conversion=force_conversion,
        property_conversion=property_conversion,
        stress_handle=stress_handle,
        stress_conversion=stress_conversion,
        detach=detach,
      )

      self.database_file = database_file
      # self.qe_calc = Espresso(pseudopotentials=pseudopotentials, tstress=tstress, tprnfor=tprnfor, kpts=kpts)

  def calculate(self, system):

    atoms = system.ase_object
    # atoms.calc = self.qe_calc
    atoms.calc = EMT()

    self.results = {
      'forces': atoms.get_forces(),
      'energy': atoms.get_potential_energy(),
    }

    if self.database_file is not None:
      db = connect(self.database_file)
      db.write(atoms)

    # for p in output:
    #   # this doesnt matter for the crystal example
    #   padded_output = torch.zeros(max_natoms, *output[p].shape[1:])
    #   padded_output[: output[p].shape[0], ...] = torch.from_numpy(
    #     output[p]
    #   )
    #   self.results[p].append(padded_output)
    #   self.results[p] = torch.stack(self.results[p]).to(system.device)

    self._update_system(system)