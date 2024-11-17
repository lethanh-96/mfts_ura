import approximator

def approximate(args):
    # input
    K_a = args.expected_n_event * 100
    K_tot = 10000
    p_a = K_a / K_tot
    # solver
    solver = approximator.create_solver(args)
    solver.run(p_a, K_a)
