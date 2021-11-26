import os
import numpy as np
import pandas as pd
import csv
import math
from PIL import Image
import time
import yaml
import argparse
import pickle
from Bio.Seq import Seq


class RNAstructure:
    def __init__(self, args):
        config = args.config
        with open(config) as yml:
            config = yaml.load(yml, yaml.FullLoader)

        data_cfg = config["DATA"]
        domain_list = [
            "cas9_wt_kim",
            "cas9_wt_wang",
            "cas9_wt_xiang",
            "cas9_hf_wang",
            "cas9_esp_wang",
        ]
        domain_file = [f"{data_cfg['in_dir']}/{data_cfg[x]}" for x in domain_list]
        target_domain = domain_list[args.target]
        
        self.png_dir = f"{data_cfg['png_dir']}/{target_domain}"
        self.load_file = domain_file[args.target]
        self.out_file = args.outfile

        self.seqlen = 33  # [42 if int(args.source) != 5 else 43, 42 if int(args.target) != 5 else 43]
        self.padding = 35  # -> 30

        self.encoding_col = "Sequence-window"
        self.scaffold = "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU"

    def Gaussian(self, x):
        return math.exp(-0.5 * (x * x))

    def adding_df(self, res, row):
        for key in row.keys():
            if key not in res:
                res[key] = list()
            res[key].append(row.get(key))

    def load_data(self):
        def refine_y(string):
            y = float(string)
            if y > 100:
                y = 100.0
            elif y < 0:
                y = 0.0
            return y

        def window_sequence(sgrna, strand, exon):
            if strand == "+":
                Sequence = exon
            else:
                Sequence = str(Seq(exon).reverse_complement())
            idx = Sequence.find(sgrna)
            if idx > -1:
                st = idx - 40  # 10 -> 40
                ed = idx + len(sgrna) + 40  # 10 -> 40
                window_size = ed - st
                window_seq = Sequence[st:ed]
                if window_size == len(Sequence[st:ed]):
                    return window_seq, Sequence
                else:
                    return "False", "False"
            else:
                "False", "False"

        res = dict()
        seen = set()
        with open(self.load_file) as fh:
            dreader = csv.DictReader(fh, delimiter="\t")
            for i, row in enumerate(dreader):
                file_name = f"{self.png_dir}/file_{i}.tiff"
                row["Y"] = refine_y(row["sgRNA-efficiency"])
                window_seq, guideRNA = window_sequence(
                    row["sgRNA-seq"], row["sgRNA-strand"], row["Sequence-target"]
                )
                if window_seq != "False":
                    if window_seq not in seen:
                        row[self.encoding_col] = window_seq[
                            self.padding : self.seqlen + self.padding
                        ]
                        seq = guideRNA.replace("T", "U") + self.scaffold
                        mat = self.rna_structure(seq)
                        im = Image.fromarray(mat)
                        im.save(file_name)

                        summary_row = {
                            "X": [char for char in row[self.encoding_col]],
                            "R": file_name,
                            "Y": row["Y"],
                        }
                        self.adding_df(res, summary_row)

        return res

    def paired(self, x, y):
        if x == "A" and y == "U":
            return 2
        elif x == "G" and y == "C":
            return 3
        elif x == "G" and y == "U":
            return 0.8
        elif x == "U" and y == "A":
            return 2
        elif x == "C" and y == "G":
            return 3
        elif x == "U" and y == "G":
            return 0.8
        else:
            return 0

    def save_dict(self, outfile, data):
        with open(outfile, "wb") as f:
            pickle.dump(data, f)

    def rna_structure(self, seq):
        # start_time = time.time()
        mat = np.zeros([len(seq), len(seq)])
        for i in range(len(seq)):
            for j in range(len(seq)):
                coefficient = 0
                for add in range(30):
                    if i - add >= 0 and j + add < len(seq):
                        score = self.paired(seq[i - add], seq[j + add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * self.Gaussian(add)
                    else:
                        break
                if coefficient > 0:
                    for add in range(1, 30):
                        if i + add < len(seq) and j - add >= 0:
                            score = self.paired(seq[i + add], seq[j - add])
                            if score == 0:
                                break
                            else:
                                coefficient = coefficient + score * self.Gaussian(add)
                        else:
                            break
                mat[i, j] = coefficient

        return mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file path")
    parser.add_argument("--target", type=int, help="Typing the target file index.")
    parser.add_argument("--outfile", type=str, help="Typing out file path and name.")
    args = parser.parse_args()

    RNAss = RNAstructure(args)
    ret = RNAss.load_data()
    RNAss.save_dict(args.outfile, ret)
