import torch.nn as nn
import torch


class NYCTaxiFareFeatureCreator(nn.Module):
    def __init__(self):
        super(NYCTaxiFareFeatureCreator, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(6, 10),    # Passenger count - 1
            nn.Embedding(7, 10),    # Year - 2009
            nn.Embedding(12, 10),   # Month
            nn.Embedding(7, 10),    # Weekday
            nn.Embedding(96, 10)    # Quaterhour
        ])

    def forward(self, x, y):
        out = [x]
        for i, v in enumerate(y.t()):
            out.append(self.embeddings[i](v))

        return torch.cat(out, dim=1)


class NYCTaxiFareModel(nn.Module):
    def __init__(self, dim_in, dim_out, embeddings=True):
        super(NYCTaxiFareModel, self).__init__()

        self.feature_creator = NYCTaxiFareFeatureCreator()

        self.layers = nn.ModuleList([
            nn.Linear(dim_in, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, dim_out),
            nn.Softmax()
        ])

    def forward(self, x, y):
        out = self.feature_creator(x, y)
        # out = torch.cat([x, y.float()], dim=1)
        for layer in self.layers:
            out = layer(out)

        return out
