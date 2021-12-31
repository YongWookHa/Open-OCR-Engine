import argparse
from pathlib import Path

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='input corpus to split into words')
    parser.add_argument('--output', type=str, required=True,
                        help='output corpus filename')
    parser.add_argument('--max_len', type=int, default=25,
                        help='max length of a word to save')
    args = parser.parse_args()

    line_cnt = 0
    out = Path(args.output_corpus).open('w', encoding='utf8')
    with Path(args.corpus).open('r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            for word in line.split():
                if args.max_len < len(word):
                    continue
                out.write(word+'\n')
                line_cnt += 1
    print(line_cnt, 'lines processed')

