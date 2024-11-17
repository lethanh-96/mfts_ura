from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class SPPP:

    def __init__(self, args):
        # save args
        self.args = args
        # compute PPP rate for devices and events
        args.lamda_d = args.expected_n_device / (np.pi * args.r_max ** 2 - np.pi * args.r_min ** 2)
        args.lamda_e = args.expected_n_event / (np.pi * args.r_max ** 2 - np.pi * args.r_min ** 2)
        # sample device location
        self.sample_device_location()

    def sample_rectangular_ppp(self, lamda, x_min, x_max, y_min, y_max):
        # get expected number of point
        expected_n_point = lamda * (x_max - x_min) * (y_max - y_min)
        # sample number of point
        n_point = np.random.poisson(expected_n_point)
        # convert them to 2d
        x = torch.rand(n_point) * (x_max - x_min) + x_min
        y = torch.rand(n_point) * (y_max - y_min) + y_min
        return x, y

    def sample_location(self, lamda, r_min, r_max):
        # sample uniform PPP in a square, super set of the ring
        x, y = self.sample_rectangular_ppp(lamda, -r_max, r_max, -r_max, r_max)
        # extract point within the ring
        r = torch.sqrt(x ** 2 + y ** 2)
        idx = torch.where(torch.logical_and(r_min <= r, r <= r_max))[0]
        x = x[idx]
        y = y[idx]
        return x, y

    def sample_device_location(self):
        # extract args
        args = self.args
        # sample device location
        self.x_d, self.y_d = self.sample_location(args.lamda_d, args.r_min, args.r_max)
        # compute distance from device to bs
        self.d = torch.sqrt(self.x_d ** 2 + self.y_d ** 2)
        # extract number of device
        n_device = len(self.x_d)
        # initialize the marginal probability that a device is triggered by an event p_x
        self.p_x = torch.zeros(n_device, dtype=torch.float)
        # initialize the markov state for each device
        self.device_state = torch.zeros(n_device, dtype=torch.float)
        # initialize active flag
        self.active = torch.zeros(n_device, dtype=torch.bool)

    def sample_event_location(self):
        # extract args
        args = self.args
        # sample event location
        self.x_e, self.y_e = self.sample_location(args.lamda_e, args.r_min, args.r_max)

    def plot(self):
        # extract args
        args = self.args
        # plot ring boundary
        Circle((0, 0), args.r_min, color='black')
        Circle((0, 0), args.r_max, color='black')
        # scatter device
        plt.scatter(self.x_d.detach().cpu().numpy(),
                    self.y_d.detach().cpu().numpy(),
                    s=0.1, c='b')
        # scatter device
        plt.scatter(self.x_e.detach().cpu().numpy(),
                    self.y_e.detach().cpu().numpy(),
                    s=10, c='r')
        # save figure
        path = os.path.join(args.figure_dir, f'{args.traffic_model}.pdf')
        plt.savefig(path)
        plt.clf()

    def get_active_device(self):
        # extract args
        args     = self.args
        r_d      = args.r_d
        n_device = len(self.x_d)
        q        = args.q
        p_ru     = args.p_ru
        # sample new event location
        self.sample_event_location()
        n_event = len(self.x_e)
        # compute the marginal probability that a device is triggered by an event p_x
        if n_event == 0:
            # set p_x as 0
            self.p_x[:]= 0.0
        else:
            # compute distance between devices and events
            d = torch.stack([torch.sqrt((self.x_e[e] - self.x_d) ** 2 + (self.y_e[e] - self.y_d) ** 2) for e in range(n_event)])
            # compute joint probability device triggered by event p_xy
            p_xy = torch.exp(- d / r_d)
            # gather marginal triggered probability
            self.p_x[:] = 1 - torch.prod(1 - p_xy, dim=0)
        # construct transition matrix
        p1 = torch.stack([1-self.p_x, self.p_x])
        p2 = torch.stack([torch.full([n_device], 1-q), torch.full([n_device], q)])
        P = torch.stack([p1, p2])
        P = P.permute(2, 0, 1)
        P = P[:, 0, :] * \
                (1 - self.device_state[:, None]) + \
                P[:, 1, :] * self.device_state[:, None]
        P = torch.cumsum(P, dim=1)
        # markov step
        r = torch.rand(n_device)
        for state_id in range(2 - 1, -1, -1):
            self.device_state[r < P[:, state_id]] = state_id
        # emission step of markov model
        self.active[:] = 0
        # devices in alarm state is always on
        idx = torch.where(self.device_state == 1)[0]
        self.active[idx] = 1
        # save the alarm probability
        # p_alarm = torch.mean(self.active).item()
        # devices in regular state is on with probability p_regular_active
        r = torch.rand(n_device)
        idx = torch.where(torch.logical_and(r < p_ru, self.device_state == 0))[0]
        self.active[idx] = 1
        # p_regular = (torch.mean(self.active) - p_alarm).item()
        # n_active = torch.sum(self.active).item()
        # print(f'[+] debug: n_alarm={int(p_alarm*n_device):d} n_regular={int(p_regular*n_device):d} n_active={n_active} K_tot={n_device}')
        return torch.nonzero(self.active, as_tuple=True)[0]
