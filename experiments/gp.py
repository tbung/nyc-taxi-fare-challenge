import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

from networks import NYCTaxiFareFeatureCreator, NYCTaxiFareModel


class GPRegressionLayer(gpytorch.models.GridInducingVariationalGP):
    def __init__(self, grid_size=10, grid_bounds=[(-1, 1)]*2):
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
    def __init__(self, feature_extractor, grid_bounds=(-1., 1.), clusters=None):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPRegressionLayer()
        self.grid_bounds = grid_bounds
        self.clusters = clusters

    def forward(self, x, y):
        features = self.feature_extractor(x, y)
        features = gpytorch.utils.scale_to_bounds(features,
                                                  self.grid_bounds[0],
                                                  self.grid_bounds[1])
        res = self.gp_layer(features)
        return res
