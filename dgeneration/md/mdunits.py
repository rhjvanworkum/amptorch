from ase import units
import numpy as np

class MDUnits:

  # internal units
  energy_unit = units.kJ / units.mol
  length_unit = units.nm
  mass_unit = 1.0  # 1 Dalton in ASE reference frame
  time_unit = length_unit * np.sqrt(mass_unit / energy_unit)

  # General utility units for conversion
  fs2internal = units.fs / time_unit
  da2internal = 1.0 / mass_unit
  angs2internal = units.Angstrom / length_unit
  bar2internal = 1e5 * units.Pascal / (energy_unit / length_unit ** 3)

  # Constants in internal units
  kB = units.kB / energy_unit  # Always uses Kelvin
  hbar = (
      units._hbar * (units.J * units.s) / (energy_unit * time_unit)
  )  # hbar is given in J/s by ASE

  @staticmethod
  def _parse_unit(unit):
    if type(unit) == str:
      # If a string is given, split into parts.
      parts = re.split("(\W)", unit.lower())

      conversion = 1.0
      divide = False
      for part in parts:
        if part == "/":
          divide = True
        elif part == "" or part == " ":
          pass
        else:
          if divide:
            conversion /= MDUnits.conversions[part]
            divide = False
          else:
            conversion *= MDUnits.conversions[part]
      return conversion
    else:
      # If unit is given as number, return number
      return unit

  @staticmethod
  def unit2internal(unit):
    conversion = MDUnits._parse_unit(unit)
    return conversion
