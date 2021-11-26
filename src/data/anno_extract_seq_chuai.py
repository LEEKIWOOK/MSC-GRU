import csv
import os


import pickle
import pyfaidx
import pyranges as pr
import yaml


class DB_LOAD:
    def __init__(self, infile: str):
        config_file = "/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/gflas-knockout-efficiency/Projects_source/transfer_efficiency_cas9/data/extract_seq.yaml"
        with open(config_file) as yml_fh:
            config_path = yaml.load(yml_fh, Loader=yaml.FullLoader)
        config = config_path["PATH"]

        # self.hg38 = f"{config['REF_DIR']}/{config['HG38_GENOME']}"
        self.in_file = f"{config['INDIR']}/{infile}"

        # self.hg38_db = pyfaidx.Fasta(self.hg38)

    def find_seq(self, rows):
        chrom = rows["Chromosome"]
        start = int(rows["Start"]) - 5
        end = int(rows["End"]) + 5

        seq = self.hg38_db[chrom][start:end]
        return str(seq)

    def save_dict(self, outfile, data):
        with open(outfile, "wb") as f:
            pickle.dump(data, f)

    def adding_df(self, res, row):
        for key in row.keys():
            if key not in res:
                res[key] = list()
            res[key].append(row.get(key))

    def load_data(self):

        res = dict()
        with open(self.in_file) as fh:
            dreader = csv.DictReader(fh, delimiter="\t")

            for i, row in enumerate(dreader):
                # seq = self.find_seq(row)
                seq = self.find_seq(row)
                seq = row["seq"]
                if len(seq) == 33:
                    row_line = {
                        "X": [char for char in seq],
                        "logY": row["Normalized_efficacy"],
                    }
                    self.adding_df(res, row_line)

        return res


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(description="raw data parser")
    PARSER.add_argument("--infile", help="input file name(preprocessed.txt)")
    PARSER.add_argument("--outfile", help="output file name(annotated.txt)")
    ARGS = PARSER.parse_args()

    FOBJ = DB_LOAD(ARGS.infile)
    ret = FOBJ.load_data()
    FOBJ.save_dict(ARGS.outfile, ret)
