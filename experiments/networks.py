import torch.nn as nn
import torch


class NYCTaxiFareFeatureCreator(nn.Module):
    def __init__(self):
        super(NYCTaxiFareFeatureCreator, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(6, 3),    # Passenger count - 1
            nn.Embedding(7, 4),    # Year - 2009
            nn.Embedding(12, 6),   # Month
            nn.Embedding(7, 4),    # Weekday
            nn.Embedding(96, 50)    # Quaterhour
        ])

    def forward(self, x, y):
        out = [x]
        for i, v in enumerate(y.t()):
            out.append(self.embeddings[i](v))

        return torch.cat(out, dim=1)


class NYCTaxiFareModel(nn.Module):
    def __init__(self, dim_in, dim_out, embeddings=True, softmax=True):
        super(NYCTaxiFareModel, self).__init__()

        self.feature_creator = NYCTaxiFareFeatureCreator()
        self.dim_out = dim_out

        self.layers = nn.ModuleList([
            nn.Linear(dim_in, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.Linear(64, dim_out),
        ])
        if softmax:
            self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        out = self.feature_creator(x[:, :4], x[:, 4:].long())
        # out = torch.cat([x, y.float()], dim=1)
        for layer in self.layers:
            out = layer(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()

        self.layers = nn.ModuleList([
            nn.BatchNorm1d(dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
        ])

        self.resample = None

        if dim_in != dim_out:
            self.resample = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        residual = x

        if self.resample:
            residual = self.resample(residual)

        for layer in self.layers:
            x = layer(x)

        return x + residual


class NYCTaxiFareResNet(nn.Module):
    def __init__(self, dim_in, dim_out, blocks, softmax=True):
        super(NYCTaxiFareResNet, self).__init__()

        self.feature_creator = NYCTaxiFareFeatureCreator()

        self.layers = nn.ModuleList([
            ResidualBlock(dim_in, 4096)
        ])

        self.layers.extend(
            [ResidualBlock(4096, 4096) for _ in range(blocks[0])]
        )

        self.layers.append(ResidualBlock(4096, 2048))
        self.layers.extend(
            [ResidualBlock(2048, 2048) for _ in range(blocks[0])]
        )

        self.layers.append(ResidualBlock(2048, 512))
        self.layers.extend(
            [ResidualBlock(512, 512) for _ in range(blocks[0])]
        )

        self.layers.append(ResidualBlock(512, 256))
        self.layers.extend(
            [ResidualBlock(256, 256) for _ in range(blocks[0])]
        )

        self.layers.append(ResidualBlock(256, dim_out))

        if softmax:
            self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        x = self.feature_creator(x[:, :4], x[:, 4:].long())
        for layer in self.layers:
            x = layer(x)

        return x
