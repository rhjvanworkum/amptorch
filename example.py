import numpy as np
import torch
from ase import Atoms
from ase.calculators.emt import EMT

from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer

from dgeneration.md.calculators.qe_calculator import QECalculator
from dgeneration.md import System, MaxwellBoltzmannInit, Simulator, VerletVelocity
from ase.lattice.cubic import FaceCenteredCubic

def gen_images():
    distances = np.linspace(2, 5, 100)
    images = []
    for dist in distances:
        image = Atoms(
            "CuCO",
            [
                (-dist * np.sin(0.65), dist * np.cos(0.65), 0),
                (0, 0, 0),
                (dist * np.sin(0.65), dist * np.cos(0.65), 0),
            ],
        )
        image.set_cell([10, 10, 10])
        image.wrap(pbc=True)
        image.set_calculator(EMT())
        images.append(image)
    return images

def gen_md_trajectory():
    md_device = "cpu"
    md_initializer = MaxwellBoltzmannInit(300, remove_rotation=False, remove_translation=False)
    md_system = System(device=md_device, initializer=md_initializer)
    md_system.load_ase_object(FaceCenteredCubic('Ni', latticeconstant=2.0))

    md_integrator = VerletVelocity(0.5)

    md_calculator = QECalculator(
        database_file="data/test-1.db",
        pseudopotentials={
            'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'
        },
        tstress=True,
        tprnfor=True,
        kpts=(3, 3, 3)
    )

    md_simulator = Simulator(md_system, md_integrator, md_calculator)
    md_simulator.simulate(100)

    return "data/test-1.db"

images = gen_images()
# dbpath = gen_md_trajectory()

Gs = {
    "default": {
        "G2": {
            "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
            "rs_s": [0],
        },
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 6,
    },
}

config = {
    "model": {
        "get_forces": True,
        "num_layers": 3,
        "num_nodes": 5,
        "batchnorm": False,
    },
    "optim": {
        "force_coefficient": 0.04,
        "lr": 1e-2,
        "batch_size": 32,
        "epochs": 100,
        "loss": "mse",
        "metric": "mae",
        "gpus": 0,
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0.1,
        "fp_params": Gs,
        "save_fps": True,
        # feature scaling to be used - normalize or standardize
        # normalize requires a range to be specified
        "scaling": {"type": "normalize", "range": (0, 1)},
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # Weights and Biases used for logging - an account(free) is required
        "logger": False,
    },
}

torch.set_num_threads(1)
trainer = AtomsTrainer(config)
trainer.train()

predictions = trainer.predict(images)

true_energies = np.array([image.get_potential_energy() for image in images])
pred_energies = np.array(predictions["energy"])

print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)))

image.set_calculator(AMPtorch(trainer))
image.get_potential_energy()
