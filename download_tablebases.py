import argparse
import os
import re
import urllib.request

URL = "http://tablebase.sesse.net/syzygy/3-4-5/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tablebases")
    parser.add_argument("--wdl-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        listing = response.read().decode("utf-8")

    extensions = ["rtbw"] if args.wdl_only else ["rtbw", "rtbz"]
    names = sorted(set(re.findall(r'href="([KQRBNP]+v[KQRBNP]+\.rtb[wz])"', listing)))
    names = [name for name in names if name.split(".")[1] in extensions]

    for i, name in enumerate(names, start=1):
        path = os.path.join(args.out, name)
        if os.path.exists(path):
            continue
        urllib.request.urlretrieve(URL + name, path + ".part")
        os.replace(path + ".part", path)
        print(f"{i}/{len(names)} {name}", flush=True)

    print(f"Done: {len(names)} files in {args.out}")


if __name__ == "__main__":
    main()
