import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, WhiteNoiseKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior, MultivariateNormalPrior
from gpytorch.random_variables import GaussianRandomVariable

import torch
import numpy as np


class SVGPRegressionLayer(gpytorch.models.AdditiveGridInducingVariationalGP):
    def __init__(self, clusters, grid_size=8, grid_bounds=[(0, 1)]):
        super(SVGPRegressionLayer, self).__init__(
            n_components=clusters.shape[0], grid_size=grid_size,
            grid_bounds=grid_bounds, mixing_params=True
        )
        self.register_parameter(
                name="mixing_params",
                parameter=torch.nn.Parameter(torch.ones(clusters.shape[0])/clusters.shape[0]),
                prior=MultivariateNormalPrior(clusters,
                                              covariance_matrix=torch.eye(clusters.shape[0])),
            )
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel(
            log_lengthscale_prior=SmoothedBoxPrior(np.exp(-3), np.exp(3), sigma=0.1, log_transform=True)
        )
        # self.noise_covar_module = WhiteNoiseKernel(variances=torch.ones(11) * 0.001)
        # self.covar_module = ScaleKernel(self.rbf_covar_module + self.noise_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, grid_bounds=(0, 1),
                 clusters=None):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = SVGPRegressionLayer(clusters)
        self.grid_bounds = grid_bounds
        self.clusters = clusters

    def forward(self, x, y):
        features = self.feature_extractor(x, y)
        # features = torch.nn.functional.softmax(features, dim=0)
        features = gpytorch.utils.scale_to_bounds(features,
                                                  self.grid_bounds[0],
                                                  self.grid_bounds[1])
        res = self.gp_layer(features)
        return res
