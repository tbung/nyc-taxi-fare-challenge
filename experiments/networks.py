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
