import os

def reset(args):
    for d in [args.figure_dir, args.csv_dir]:
        if os.path.exists(d):
            os.system(f'rm -rf {d}')
