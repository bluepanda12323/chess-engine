import argparse
import struct

import torch

from ai.model import ValueModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/value_model.tar")
    parser.add_argument("--out", default="models/value_weights.bin")
    args = parser.parse_args()

    model = ValueModel(num_inputs=448, num_hidden=400, num_layers=5)
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location="cpu"))

    layers = list(model.hidden) + [model.output]
    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", len(layers)))
        for layer in layers:
            out_features, in_features = layer.weight.shape
            f.write(struct.pack("<II", out_features, in_features))
            f.write(layer.weight.detach().numpy().astype("<f4").tobytes())
            f.write(layer.bias.detach().numpy().astype("<f4").tobytes())

    print(f"{len(layers)} layers -> {args.out}")


if __name__ == "__main__":
    main()
