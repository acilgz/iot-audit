import argparse
from data_loading import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/train_test_network.csv')
    ap.add_argument('--outdir', default='train_mc')
    ap.add_argument('--models', nargs='+', choices=['rf_mc', 'lgbm_mc', 'xgb_mc', 'logreg_mc', 'mlp_mc'], default=['rf_mc', 'lgbm_mc', 'xgb_mc', 'logreg_mc', 'mlp_mc'])
    args = ap.parse_args()
    prepare(args.csv, args.outdir, list(dict.fromkeys(args.models)), 'multiclass')


if __name__ == '__main__':
    main()
