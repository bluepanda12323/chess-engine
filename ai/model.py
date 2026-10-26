from torch import nn


class ChessModel(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_layers, num_outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.ModuleList()
        in_features = num_inputs
        for _ in range(num_layers):
            self.hidden.append(nn.Linear(in_features=in_features, out_features=num_hidden))
            in_features = num_hidden
        self.act_fn = nn.ReLU()
        self.output = nn.Linear(in_features=num_hidden, out_features=num_outputs)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.hidden:
            x = self.act_fn(layer(x))
        x = self.output(x)
        return x


class ValueModel(ChessModel):

    def __init__(self, num_inputs, num_hidden, num_layers):
        super().__init__(num_inputs, num_hidden, num_layers, num_outputs=1)
        self.output_act_fn = nn.Tanh()

    def forward(self, x):
        x = super().forward(x)
        x = self.output_act_fn(x)
        return x[:, 0]
