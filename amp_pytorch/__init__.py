"""Base AMP calculator class to be utilized similiar to other ASE calculators"""

import copy
import numpy as np
import os
from torch.utils.data import DataLoader
from amp.utilities import Logger
from amp.descriptor.gaussian import Gaussian
from amp_pytorch.data_preprocess import (
    AtomsDataset,
    TestDataset,
    factorize_data,
    collate_amp,
)
from amp_pytorch.NN_model import FullNN, CustomLoss
from amp_pytorch.trainer import Trainer
from ase.calculators.calculator import Calculator, Parameters
import torch

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class AMP(Calculator):
    """Atomistics Machine-Learning Potential (AMP) ASE calculator
   Parameters
   ----------
    model : object
    Class representing the regression model. Input arguments include training
    images and descriptor type. Model structure and training schemes can be
    modified directly within the class.

    label : str
    Location to save trained the trained model.

    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, label="amptorch.pt"):
        Calculator.__init__(self)

        self.model = model
        self.label = label
        self.fp_scaling = self.model.training_data.fprange
        self.target_sd = self.model.training_data.scaling_sd
        self.target_mean = self.model.training_data.scaling_mean

    def train(self, overwrite=False):

        self.trained_model = self.model.train()
        if os.path.exists(self.label):
            if overwrite is False:
                print("Could not save! File already exists")
            else:
                torch.save(self.trained_model.state_dict(), self.label)
        else:
            torch.save(self.trained_model.state_dict(), self.label)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        dataset = TestDataset(atoms, self.model.descriptor, self.fp_scaling)
        fp_length = dataset.fp_length()
        unique_atoms = dataset.unique()
        architecture = copy.copy(self.model.structure)
        architecture.insert(0, fp_length)
        batch_size = len(dataset)
        dataloader = DataLoader(
            dataset, batch_size, collate_fn=dataset.collate_test, shuffle=False
        )
        model = FullNN(unique_atoms, architecture, "cpu")
        model.load_state_dict(torch.load("amptorch.pt"))

        for batch in dataloader:
            input_data = [batch[0], len(batch[1])]
            for element in unique_atoms:
                input_data[0][element][0] = input_data[0][element][0].requires_grad_(
                    True
                )
            fp_primes = batch[2]
            energy, forces = model(input_data, fp_primes)
        energy = (energy * self.target_sd) + self.target_mean
        forces = forces * self.target_sd

        self.results["energy"] = np.concatenate(energy.detach().numpy())
        self.results["forces"] = forces.detach().numpy()
