import torch
import pyro.contrib.gp as gp


class DGP:
    def __init__(self, n_layers, Xu, n_data):
        self.layers = []
        self.device = 'cuda'

        kernel = gp.kernels.RBF(
            input_dim=9, variance=torch.tensor(2.0),
            lengthscale=torch.tensor(2.0)
        ).add(gp.kernels.WhiteNoise(input_dim=9))

        likelihood = gp.likelihoods.Gaussian(variance=torch.tensor(0.01))

        self.layers.append(gp.models.VariationalSparseGP(
            X=Xu, y=None, kernel=kernel, Xu=Xu, likelihood=likelihood,
            num_data=n_data, whiten=True, name="SVGP0", latent_shape=(9,),
            mean_function=lambda x: x.t()
        ).to(self.device))

        for i in range(1, n_layers-1):
            kernel = gp.kernels.RBF(
                input_dim=9, variance=torch.tensor(2.0),
                lengthscale=torch.tensor(2.0)
            ).add(gp.kernels.WhiteNoise(input_dim=9))

            likelihood = gp.likelihoods.Gaussian(variance=torch.tensor(0.01))

            self.layers.append(gp.models.VariationalSparseGP(
                X=self.layers[-1].model()[0].t(), y=None, kernel=kernel, Xu=Xu,
                likelihood=likelihood, num_data=n_data,
                whiten=True, name=f"SVGP{i}", latent_shape=(9,),
                mean_function=lambda x: x.t()
            ).to(self.device))

        kernel = gp.kernels.RBF(
            input_dim=9, variance=torch.tensor(2.0),
            lengthscale=torch.tensor(2.0)
        ).add(gp.kernels.WhiteNoise(input_dim=9))

        likelihood = gp.likelihoods.Gaussian(variance=torch.tensor(0.01))

        self.layers.append(gp.models.VariationalSparseGP(
            X=self.layers[-1].model()[0].t(), y=None, kernel=kernel, Xu=Xu,
            likelihood=likelihood, num_data=n_data,
            whiten=True, name=f"SVGP{n_layers-1}"
        ).to(self.device))

    def model(self, x, y):
        self.layers[0].set_data(x)
        for i, layer in enumerate(self.layers[1:-1]):
            layer.set_data(self.layers[i].model()[0].t())

        self.layers[-1].set_data(self.layers[-2].model()[0].t(), y)

        return self.layers[-1].model()

    def guide(self, x, y):
        self.layers[0].set_data(x)
        for i, layer in enumerate(self.layers[1:-1]):
            layer.set_data(self.layers[i].model()[0].t())

        self.layers[-1].set_data(self.layers[-2].model()[0].t(), y)

        return self.layers[-1].guide()

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x, _ = layer(x)
            x = x.t()

        return self.layers[-1](x)

    def to(self, device):
        for layer in self.layers:
            layer.to(device)

    def cpu(self):
        self.to('cpu')
