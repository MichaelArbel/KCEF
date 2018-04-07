import time
import numpy as np


def ancestral_hmc(transition_kernel, momentums, graph, num_samples, num_iter, recompute_log_pdf=False, time_budget=None, X_cond= None):
    # MCMC results
    D = 0
    
    i0 = 0
    sample_coordinates = []
    for node in graph:
        tmp_d = max(node[0])
        sample_coordinates = sample_coordinates + node[0]
        if tmp_d > D:
            D = tmp_d
    D = D+1

    num_nodes = len(graph)

    samples = np.random.rand(num_samples, D) + np.nan

    accepted = np.zeros((num_nodes)) + np.nan
    acc_prob = np.zeros(( num_nodes)) + np.nan
    log_pdf = np.zeros(( num_nodes)) + np.nan
    
    # timings for output and time limit
    times = np.zeros(num_samples)
    last_time_printed = time.time()
    
    # for adaptive transition kernels
    avg_accept = 0.
    
    for d, node in enumerate(graph):
            # init MCMC (first iteration)
        d_node = len(node[0])
        current = np.random.rand(num_samples, len(node[0]))
        current_log_pdf = transition_kernel.target.log_pdf(current)
        tmp_samples = np.zeros((num_samples, d_node)) + np.nan
        tmp_log_pdf = np.zeros((num_samples, d_node)) + np.nan
        tmp_proposals = np.zeros((num_samples, d_node)) + np.nan
        tmp_accepted = np.zeros((num_samples, d_node)) + np.nan
        tmp_acc_prob    = np.zeros(num_samples) + np.nan
        tmp_times      = np.zeros(num_iter)

        last_time_printed = time.time()

        X_cond = np.array([samples[:,i] for i in graph[d][1]]).T

        transition_kernel.target.set_cond(X_cond, d )

        transition_kernel.momentum = momentums[d]

        for it in range(num_iter):
            tmp_times[it] = time.time()
                            
            # marginal sampler: make transition kernel re-compute log_pdf of current state
            if recompute_log_pdf:
                current_log_pdf = None
            
            tmp_proposals, tmp_acc_prob, log_pdf_proposal = transition_kernel.proposal(current, current_log_pdf)
            
            # accept-reject
            r = np.random.rand(num_samples)
            tmp_accepted = r < tmp_acc_prob
                        
            
            # update running mean according to knuth's stable formula
            avg_accept += (tmp_accepted - avg_accept) / (it + 1)
            
            # update state

            #transition_kernel.target.reset()
            current[tmp_accepted] = tmp_proposals[tmp_accepted]
            current_log_pdf[tmp_accepted] = 1.*log_pdf_proposal[tmp_accepted]
            # store sample
            tmp_samples = current
            tmp_log_pdf = current_log_pdf

            # store step size
            #print ('iteration %d ' % it)    
        index = 0
        #print tmp_samples
        #tmp_samples = np.reshape(tmp_samples, [num_samples,-1])
        for i in graph[d][0]:  
            samples[:,i-i0] = tmp_samples[:,index]
            index += 1

    # recall it might be less than last iterations due to time budget
    samples = samples[:,sample_coordinates]
    return samples
