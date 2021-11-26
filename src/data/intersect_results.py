import csv
import pandas as pd
import pickle


class DB_LOAD:
    def __init__(self, infile1: str, infile2: str):
        self.infile1 = infile1
        self.infile2 = infile2

    def merged_data(self):

        off_data = pd.read_csv(
            self.infile1, sep="\t", skiprows=2, usecols=[3], names=["window-seq"]
        )
        eff_data = pd.read_csv(
            self.infile2,
            sep="\t",
            skiprows=1,
            usecols=[4, 5],
            names=["seq", "efficiency"],
        )

        off_data["seq"] = [x[5:28] for x in off_data["window-seq"]]
        merged = pd.merge(off_data, eff_data, how="inner", on="seq")

        res = dict()
        for index, row in merged.iterrows():
            row_line = {
                "X": [ch for ch in row["window-seq"]],
                "Y": row["efficiency"],
            }
            self.adding_df(res, row_line)

        return res

    def save_dict(self, outfile, data):
        with open(outfile, "wb") as f:
            pickle.dump(data, f)

    def adding_df(self, res, row):
        for key in row.keys():
            if key not in res:
                res[key] = list()
            res[key].append(row.get(key))


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(description="raw data parser")
    PARSER.add_argument("--infile1", help="input file name(cas-offinder)")
    PARSER.add_argument("--infile2", help="input file name(efficiency.tsv)")
    PARSER.add_argument("--outfile", help="output file name")
    ARGS = PARSER.parse_args()

    FOBJ = DB_LOAD(ARGS.infile1, ARGS.infile2)
    ret = FOBJ.merged_data()
    FOBJ.save_dict(ARGS.outfile, ret)
