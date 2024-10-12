'''
Simple script to transform JWS data to format accepted by gensim.models.Word2Vec.wv.evaluate_word_pairs

Usage:
    preprocess_JWS_data.py <dir>
'''
import csv
import docopt
from glob import glob
import os

def transform_JWS_file(path:os.PathLike):
    data = []
    with open(path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        for line in reader:
            data.append(line[:3])
    output_path = path.removesuffix('.csv')+'.tsv'
    with open(output_path, "w", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
        writer.writerows(data)

if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    dir = args['<dir>']
    files = glob(os.path.join(dir, "score_*.csv"))
    for f in files:
        print(f)
        transform_JWS_file(f)