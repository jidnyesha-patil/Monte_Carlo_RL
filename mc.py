#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    # action
    if score >= 20: 
        action = 0
    else:
        action = 1
    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop each episode
    for every_ep in range(n_episodes):
        # initialize the episode
        obs = env.reset()
        # generate empty episode list
        episode_list = []
        # loop until episode generation is done
        done = False
        while done != True:
            # select an action
            action = initial_policy(obs)
            # return a reward and new state
            new_obs, reward,done,_= env.step(action) # New observation = Next State
            # append state, action, reward to episode
            episode_list.append([obs,action,reward])
            # update state to new state
            obs = new_obs

        this_G=0
        states_visited = []
        # loop for each step of episode, t = T-1, T-2,...,0
        for ind,this_tuple in enumerate(episode_list):
            # compute G
            this_G = sum([tup[2]*gamma**i for i,tup in enumerate(episode_list[ind:])])
            # unless state_t appears in states
            if this_tuple[0] not in states_visited:
                states_visited.append(this_tuple[0])
                # update return_count
                returns_count[this_tuple[0]]+=1 
                # update return_sum
                returns_sum[this_tuple[0]]+= this_G
                # calculate average return for this state over all sampled episodes
                V[this_tuple[0]]=returns_sum[this_tuple[0]]/returns_count[this_tuple[0]]
    ############################
    return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # Initialise all actions having probability zero
    action_probabilities = np.zeros(nA)
    # Choose greedy action and make its probability (1-epsilon)
    greedy_index = np.argmax(Q[state])
    action_probabilities[greedy_index]=1-epsilon
    # Make all other probabilities as (epsilon/nA)
    action_probabilities+=(epsilon/nA)
    action = np.random.choice(nA, p = action_probabilities)
    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n
    ############################
    # YOUR IMPLEMENTATION HERE #
    for episode_i in range(n_episodes):
        # OPTIONAL - define decaying epsilon
        #### Tests runs for 208.746 secs WITHOUT decaying epsilon
        #### Tests runs for 208.689 secs WITH decaying epsilon
        if episode_i != 0:
            epsilon -= (0.1/n_episodes)

        # initialize the episode
        obs = env.reset()
        # generate empty episode list
        episode_list =[]
        # loop until one episode generation is done
        done = False
        while done != True:
            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q,obs,nA,epsilon=epsilon)
            # return a reward and new state
            new_state,reward,done,_ = env.step(action)
            # append state, action, reward to episode
            episode_list.append([obs,action,reward])
            # update state to new state
            obs = new_state

        state_action_visited = []
        this_G = 0
        # loop for each step of episode, t = T-1, T-2, ...,0
        for ind,tuple_i in enumerate(episode_list):
            # compute G
            this_G = sum([tup[2]*gamma**i for i,tup in enumerate(episode_list[ind:])])
            # unless the pair state_t, action_t appears in <state action> pair list
            if (tuple_i[0],tuple_i[1]) not in state_action_visited:
                state_action_visited.append((tuple_i[0],tuple_i[1]))
                # update return_count
                returns_count[(tuple_i[0],tuple_i[1])]+=1
                # update return_sum
                returns_sum[(tuple_i[0],tuple_i[1])]+=this_G
                # calculate average return for this state over all sampled episodes
                Q[tuple_i[0]][tuple_i[1]] = returns_sum[(tuple_i[0],tuple_i[1])]/returns_count[(tuple_i[0],tuple_i[1])]
  
    return Q

################################# RESULTS ##########################################
# ------On-policy Monte Carlo(50 points in total)------ ... ok
# initial_policy (2 points) ... ok
# mc_prediction (20 points) ... ok
# epsilon_greedy (8 points) ... ok
# mc_control_epsilon_greedy (20 points) ... ok

# ----------------------------------------------------------------------
# Ran 5 tests in 208.746s

# OK
# ============================ With Decaying Epsilon ====================================
# ------On-policy Monte Carlo(50 points in total)------ ... ok
# initial_policy (2 points) ... ok
# mc_prediction (20 points) ... ok
# epsilon_greedy (8 points) ... ok
# mc_control_epsilon_greedy (20 points) ... ok

# ----------------------------------------------------------------------
# Ran 5 tests in 208.689s

# OK