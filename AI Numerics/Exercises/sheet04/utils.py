import numpy as np
from scipy.special import softmax

def gen_markov_simple(number_states=5, T=100, P_true=None):

    # If no transition matrix is provided, make a random one
    if P_true is None:
        P_true = np.random.rand(number_states,number_states)
        P_true /= P_true.sum(0)

    # state affiliation of X
    pi_X = np.zeros((number_states, T))

    # Initial random affiliation
    pi_X[np.random.randint(number_states),0] = 1
    
    # Iterate
    for t in range(1,T):
        probs = P_true @ pi_X[:,t-1]
        next = np.random.choice(number_states, p=probs.flatten())
        pi_X[next,t] = 1

    Xt = np.argmax(pi_X,0)

    return pi_X, Xt, P_true
    
def gen_bayes(K,M,T, deterministic=True):
    pi_X = np.zeros((K,T))
    states_X = np.random.choice(K,T)
    for k in range(K):
        pi_X[k,states_X==k] = 1
    
    Lambda_true = np.random.rand(M,K)
    if deterministic:
        Lambda_true = softmax(Lambda_true*1e5,0)
    else:
        Lambda_true += 1e-6
        Lambda_true /= Lambda_true.sum(0)
        
    prob_Y = Lambda_true @ pi_X

    pi_Y = np.zeros((M,T))
    for t in range(T):
        realization = np.random.choice(M, p=prob_Y[:,t])
        pi_Y[realization,t] = 1
    
    return pi_X,pi_Y,Lambda_true
    
