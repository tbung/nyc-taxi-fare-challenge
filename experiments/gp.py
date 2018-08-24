import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable


class NYCTaxiFareDeepKernel(nn.Module):
    def __init__(self, dims):
        super(NYCTaxiFareDeepKernel, self).__init__()

        self.layers = [
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(),
            nn.Linear(dims[3], dims[4]),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class GPRegressionLayer(gpytorch.models.GridInducingVariationalGP):
    def __init__(self, grid_size=20, grid_bounds=[(-1, 1), (-1, 1)]):
        super(GPRegressionLayer, self).__init__(grid_size=grid_size, grid_bounds=grid_bounds)
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        self.register_parameter(
            name="log_outputscale",
            parameter=nn.Parameter(torch.tensor([0]))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) * self.log_outputscale.exp()
        return GaussianRandomVariable(mean_x, covar_x)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, n_features, grid_bounds=(-1., 1.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPRegressionLayer()
        self.grid_bounds = grid_bounds
        self.n_features = n_features

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        res = self.gp_layer(features)
        return res

model = DKLModel(feature_extractor, n_features=num_features).cuda()
likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()

optimizer = torch.optim.SGD([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Our loss object. We're using the VariationalMarginalLogLikelihood, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, model, n_data=train_y.size(0))

for i in range(num_epochs):
    # Within each iteration, we will go over each minibatch of data
    for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Because the grid is relatively small, we turn off the Toeplitz matrix multiplication and just perform them directly
        # We find this to be more efficient when the grid is very small.
        with gpytorch.settings.use_toeplitz(False), gpytorch.beta_features.diagonal_correction():
            output = model(x_batch)
            loss = -mll(output, y_batch)
            print('Epoch %d [%d/%d] - Loss: %.3f' % (i + 1, minibatch_i, len(train_loader), loss.item()))

        # The actual optimization step
        loss.backward()
        optimizer.step()
