import pandas as pd
import tqdm
import os

class Monitor:

    def __init__(self, args, progress_bar=True):
        # save
        self.args = args
        # initialize progress bar
        if progress_bar:
            self.bar = tqdm.tqdm(range(args.n_step))
            self.global_step = 0
        # initialize csv
        if args.csv:
            self.csv_data = {}
        # skip runned instances
        if args.skip:
            self.skip()

    @property
    def label(self):
        args = self.args
        # label = f'{args.solver}_{args.seed}'
        # label = f'{args.rx_power_avg:0.1f}_{args.rx_power_gap:0.1f}_{args.expected_n_event:0.1f}'
        # label = f'{args.rx_power_avg}_{args.expected_n_event:0.1f}'
        # full experiment
        label = f'{args.solver}_{args.expected_n_device:0.1f}_{args.expected_n_event:0.2f}_{args.seed}'
        return label

    def update_time(self):
        self.bar.update(1)

    def update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            if 'avg_reward' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'n_active' in key:
                _kwargs[key] = kwargs[key]
            elif 'n_violated' in key:
                _kwargs[key] = kwargs[key]
            elif 'avg_p_tx' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'loss' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
        self.bar.set_postfix(**_kwargs)

    def display(self):
        self.bar.display()

    def update_csv(self, info):
        # extract args
        args = self.args
        data = self.csv_data
        Q    = args.n_arm
        # create fields if not existed
        fields = list(info.keys())
        for field in fields:
            if field not in data:
                data[field] = []
        # append data to fields
        for field in fields:
            data[field].append(info[field])

    def write_csv(self):
        # extract args
        args = self.args
        if args.csv:
            data = self.csv_data
            # convert to pandas and write to csv
            df = pd.DataFrame(data)
            path = os.path.join(args.csv_dir, f'{self.label}.csv')
            df.to_csv(path, index=None)

    def skip(self):
        # extract args
        args = self.args
        #
        path = os.path.join(args.csv_dir, f'{self.label}.csv')
        try:
            df = pd.read_csv(path)
            if len(df) >= args.n_step:
                print('skipping')
                exit()
        except:
            print('running')

    def step(self, info):
        # update progress bar
        if not self.args.headless:
            self.update_time()
            self.update_description(**info)
            self.display()
        # log to csv
        if self.args.csv:
            self.update_csv(info)
        self.global_step += 1
