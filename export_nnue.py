import argparse
import struct

import numpy as np

ENGINE_PARTS = ["feature_weights", "feature_bias", "psqt", "hidden_weight", "hidden_bias", "output_weight", "output_bias"]


def fold_weights(weights):
    bucket_weights = weights["bucket_weights"]
    num_buckets, _, width = bucket_weights.shape
    num_hidden = width - 1
    counts = np.repeat(weights["count_weights"], 64, axis=1).reshape(768, width)
    pieces = bucket_weights + weights["shared_weights"] + counts
    extra = [weights[name] for name in ("pawn_weights", "piece_count_weights", "mobility_weights") if name in weights]
    table = np.concatenate([pieces.reshape(-1, width)] + extra)
    num_layer2 = weights["hidden.weight"].shape[0] if "hidden.weight" in weights else 0
    return {
        "num_buckets": num_buckets,
        "mirror": 1,
        "pawn_features": 1,
        "count_features": int("piece_count_weights" in weights),
        "mobility_features": int("mobility_weights" in weights),
        "num_hidden": num_hidden,
        "num_layer2": num_layer2,
        "buckets": weights["bucket_table"],
        "feature_weights": table[:, :num_hidden],
        "feature_bias": weights["feature_bias"],
        "psqt": table[:, num_hidden],
        "hidden_weight": weights["hidden.weight"] if num_layer2 else None,
        "hidden_bias": weights["hidden.bias"] if num_layer2 else None,
        "output_weight": weights["output.weight"],
        "output_bias": weights["output.bias"],
    }


def dense_weights(weights):
    num_hidden = weights["feature_transformer.weight"].shape[0]
    num_layer2 = weights["hidden.weight"].shape[0] if "hidden.weight" in weights else 0
    return {
        "num_buckets": 1,
        "mirror": 0,
        "pawn_features": 0,
        "count_features": 0,
        "mobility_features": 0,
        "num_hidden": num_hidden,
        "num_layer2": num_layer2,
        "buckets": np.zeros(64),
        "feature_weights": weights["feature_transformer.weight"].T,
        "feature_bias": weights["feature_transformer.bias"],
        "psqt": weights["psqt.weight"][0] if "psqt.weight" in weights else np.zeros(768),
        "hidden_weight": weights["hidden.weight"] if num_layer2 else None,
        "hidden_bias": weights["hidden.bias"] if num_layer2 else None,
        "output_weight": weights["output.weight"],
        "output_bias": weights["output.bias"],
    }


def write_nnue(net, path):
    with open(path, "wb") as f:
        header = [net["num_buckets"], net["mirror"], net["pawn_features"], net["num_hidden"], net["num_layer2"]]
        if net["count_features"] or net["mobility_features"]:
            f.write(b"NNU3")
            f.write(struct.pack("<7I", *header, net["count_features"], net["mobility_features"]))
        else:
            f.write(b"NNU2")
            f.write(struct.pack("<5I", *header))
        f.write(np.asarray(net["buckets"], dtype="<u4").tobytes())
        for name in ENGINE_PARTS:
            if net[name] is not None:
                f.write(np.asarray(net[name], dtype="<f4").tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/nnue_weights.npz")
    parser.add_argument("--out", default="models/nnue.bin")
    args = parser.parse_args()

    weights = np.load(args.model)
    net = fold_weights(weights) if "bucket_weights" in weights else dense_weights(weights)
    write_nnue(net, args.out)

    layers = f"{net['num_hidden']} x 2 -> {net['num_layer2']} -> 1" if net["num_layer2"] else f"{net['num_hidden']} x 2 -> 1"
    pawns = "with" if net["pawn_features"] else "without"
    mobility = "with" if net["mobility_features"] else "without"
    print(f"NNUE: {net['num_buckets']} king bucket(s), {pawns} pawn structure, {mobility} counts and mobility, {layers} -> {args.out}")


if __name__ == "__main__":
    main()
