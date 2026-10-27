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


class ValueCNN(nn.Module):

    def __init__(self, num_channels, num_filters, num_conv_layers, num_hidden, num_layers):
        super().__init__()
        self.conv = nn.ModuleList()
        in_channels = num_channels
        for _ in range(num_conv_layers):
            self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1))
            in_channels = num_filters
        self.flatten = nn.Flatten()
        self.hidden = nn.ModuleList()
        in_features = num_filters * 8 * 8
        for _ in range(num_layers):
            self.hidden.append(nn.Linear(in_features=in_features, out_features=num_hidden))
            in_features = num_hidden
        self.act_fn = nn.ReLU()
        self.output = nn.Linear(in_features=num_hidden, out_features=1)
        self.output_act_fn = nn.Tanh()

    def forward(self, x):
        for layer in self.conv:
            x = self.act_fn(layer(x))
        x = self.flatten(x)
        for layer in self.hidden:
            x = self.act_fn(layer(x))
        x = self.output(x)
        x = self.output_act_fn(x)
        return x[:, 0]
