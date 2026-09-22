import argparse
from data_loading import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/train_test_network.csv')
    ap.add_argument('--outdir', default='train')
    ap.add_argument('--models', nargs='+', choices=['rf', 'lgbm', 'xgb', 'logreg', 'mlp'], default=['rf', 'lgbm', 'xgb', 'logreg', 'mlp'])
    args = ap.parse_args()
    prepare(args.csv, args.outdir, list(dict.fromkeys(args.models)), 'binary')


if __name__ == '__main__':
    main()
