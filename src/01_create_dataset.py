import argparse
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lib.opus_converter import convert_opus

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler_c = logging.StreamHandler()
handler_f = logging.FileHandler("./log/01_create_dataset.log")

handler_c.setLevel(logging.DEBUG)
handler_f.setLevel(logging.DEBUG)

format_c = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
format_f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler_c.setFormatter(format_c)
handler_f.setFormatter(format_f)

logger.addHandler(handler_c)
logger.addHandler(handler_f)

def create_dataset(dirs_in, labels):
    logger.info("Loading data")
    dirs_in = [Path(dir) for dir in dirs_in]
    if not labels:
        labels = [dir.name for dir in dirs_in]

    if len(dirs_in) != len(labels):
        raise ValueError("Directories and labels must have the same number of entries.")

    data = []
    wns = []
    lab = []

    for i, dir in enumerate(dirs_in):
        logger.info(f"Loading files from {dir}")
        files = sorted([x for x in dir.iterdir()])

        for file in files:
            logger.debug(f"Loading file {file}")
            lab.append(labels[i])
            if file.suffix.lower() == ".txt" or file.suffix.lower() == ".csv":
                filedata = np.genfromtxt(file, delimiter=",")
            elif file.suffix == ".tsv":
                filedata = np.genfromtxt(file, delimiter="\t")
            elif re.match(r"\.\d+$", file.suffix):
                filedata = convert_opus(file)
            else:
                logger.error(
                    f"Unsupported filetype for file {file}. Use csv or tsv plain text files or binary OPUS files.")
                logger.warning(f"Skipping file {file} ...")

            wns.append(filedata[:, 0])
            data.append(filedata[:, 1])



    if not all([np.array_equal(element, wns[0]) for element in wns]):
        raise ValueError("Wavenumber values are not the same in all files.")
    wns = wns[0]
    data = np.asarray(data)

    data = pd.DataFrame(data, columns=wns)
    data.insert(0, "label", lab)

    logger.info("Finished loading data.")
    return data


if __name__ == "__main__":
    logger.info("Started dataset creation")
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

    logger.info("Parsing arguments")
    args = parser.parse_args()
    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    dataset = create_dataset(args.dir, args.label)
    logger.info(f"Saving data to file {args.out[0]}")

    dataset.to_csv(args.out[0], index=False)
