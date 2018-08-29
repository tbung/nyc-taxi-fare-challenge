import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

from networks import NYCTaxiFareFeatureCreator


class NYCTaxiFareDeepKernel(nn.Module):
    def __init__(self, dims):
        super(NYCTaxiFareDeepKernel, self).__init__()
        self.feature_creator = NYCTaxiFareFeatureCreator()

        self.layers = [
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(),
            nn.Linear(dims[3], dims[4]),
        ]

    def forward(self, x, y):
        out = self.feature_creator(x, y)
        for layer in self.layers:
            out = layer(out)

        return out


class GPRegressionLayer(gpytorch.models.GridInducingVariationalGP):
    def __init__(self, grid_size=20, grid_bounds=[(-1, 1), (-1, 1)]):
        super(GPRegressionLayer, self).__init__(
            grid_size=grid_size, grid_bounds=grid_bounds
        )
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class DKLModel(gpytorch.Module):
    def __init__(self, grid_bounds=(-1., 1.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = NYCTaxiFareDeepKernel([54, 1000, 500, 50, 2])
        self.gp_layer = GPRegressionLayer()
        self.grid_bounds = grid_bounds

    def forward(self, x, y):
        features = self.feature_extractor(x, y)
        features = gpytorch.utils.scale_to_bounds(features,
                                                  self.grid_bounds[0],
                                                  self.grid_bounds[1])
        res = self.gp_layer(features)
        return res
