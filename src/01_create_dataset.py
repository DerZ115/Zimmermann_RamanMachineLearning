import argparse
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lib.opus_converter import convert_opus



def create_dataset(dirs_in, labels):
    dirs_in = [Path(dir) for dir in dirs_in]
    if not labels:
        labels = [dir.name for dir in dirs_in]

    if len(dirs_in) != len(labels):
        raise ValueError("Directories and labels must have the same number of entries.")

    data = []
    wns = []
    lab = []

    for i, dir in enumerate(dirs_in):
        files = sorted([x for x in dir.iterdir()])

        for file in files:
            lab.append(labels[i])
            if file.suffix.lower() == ".txt" or file.suffix.lower() == ".csv":
                filedata = np.genfromtxt(file, delimiter=",")
            elif file.suffix == ".tsv":
                filedata = np.genfromtxt(file, delimiter="\t")
            elif re.match(r"\.\d+$", file.suffix):
                filedata = convert_opus(file)
            else:
                raise ValueError(
                    "Unsupported filetype. Use csv or tsv-style plain text files or binary OPUS files.")

            wns.append(filedata[:, 0])
            data.append(filedata[:, 1])

    if not all([np.array_equal(element, wns[0]) for element in wns]):
        raise ValueError("Wavenumber values are not the same in all files.")
    wns = wns[0]
    data = np.asarray(data)

    data = pd.DataFrame(data, columns=wns)
    data.insert(0, "label", lab)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a single csv file from individual Raman spectra."
    )

    parser.add_argument("-d", "--dir", metavar="PATH", type=str, nargs="+", action="store",
                        help="Directories containing the Raman spectra as csv files. Each directory should contain one class.", required=True)
    parser.add_argument("-l", "--label", metavar="NAME", type=str, nargs="+", action="store",
                        help="Labels for the classes. Must have the same number of entries as '--dir'. If not provided, the directory names will be used.", required=False)
    parser.add_argument("-n", "--num", metavar="INT", type=int, nargs=1, action="store", help="Number of samples/spectra to use for each class. If 0, all samples will be kept.", default=0)
    parser.add_argument("-o", "--out", metavar="PATH", type=str, nargs=1,
                        action="store", help="Output path for the merged csv file", required=True)

    args = parser.parse_args()

    dataset = create_dataset(args.dir, args.label)

    dataset.to_csv(args.out[0], index=False)
