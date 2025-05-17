"""
Replace each RuntimeError with code that does what's
specified in the docstring, then submit to autograder.
"""

import numpy as np


def utility_gradients(logit, reward):
    """
    Calculate partial derivatives of expected rewards with respect to logits.

    @param:
    logit - player i plays move 1 with probability 1/(1+exp(-logit[i]))
    reward - reward[i,a,b] is reward to player i if player 0 plays a, and player 1 plays b

    @return:
    gradients - gradients[i]= dE[reward[i,:,:]]/dlogit[i]
    utilities - utilities[i] = E[reward[i,:,:]]
      where the expectation is computed over the distribution of possible moves by both players.
    """
    logit = np.asarray(logit)

    exp_logit = np.exp(logit)

    pa = np.array([1 / (1 + exp_logit[0]), exp_logit[0] / (1 + exp_logit[0])])
    pb = np.array([1 / (1 + exp_logit[1]), exp_logit[1] / (1 + exp_logit[1])])

    da = exp_logit[0] / (1 + exp_logit[0]) ** 2 * np.array([-1, 1])
    db = exp_logit[1] / (1 + exp_logit[1]) ** 2 * np.array([-1, 1])

    gradients, utilities = np.zeros(2), np.zeros(2)
    gradients[0] = da.T @ reward[0] @ pb
    gradients[1] = pa.T @ reward[1] @ db

    utilities[0] = pa.T @ reward[0] @ pb
    utilities[1] = pa.T @ reward[1] @ pb

    return gradients, utilities


def strategy_gradient_ascent(logit, reward, nsteps, learningrate):
    """
    nsteps of a 2-player, 2-action episodic game, strategies learned
    using simultaneous gradient ascent.

    @param:
    logit - intial logits for the two players
    reward - reward[i,a,b] is reward to player i if player 0 plays a, and player 1 plays b
    nsteps - number of steps of ascent to perform
    learningrate - learning rate

    @return:
    path - path[t,i] is the logit of the i'th player's strategy after t steps of
      simultaneous gradient ascent (path[0,:]==logit).
    utilities (nsteps,2) - utilities[t,i] is the expected reward to player i on step t,
      where expectation is over the distribution of moves given by logits[t,:]
    """
    logit = np.asarray(logit).copy()

    path = np.zeros((nsteps, 2))
    all_utilities = np.zeros((nsteps, 2))

    path[0] = logit
    gradients, all_utilities[0] = utility_gradients(logit, reward)

    curr_logit = logit.copy()
    for i in range(1, nsteps):
        gradients, utilities = utility_gradients(curr_logit, reward)

        curr_logit = curr_logit + learningrate * gradients
        path[i] = curr_logit
        all_utilities[i] = utilities

    return path, all_utilities


def mechanism_gradient(logit, reward):
    """
    Calculate partial derivative of mechanism loss with respect to rewards.

    @param:
    logit - The goal is to make this pair of strategies a Nash equlibrium:
        player i plays move 1 with probability 1/(1+exp(-logit[i])), else move 0
    reward - reward[i,a,b] is reward to player i if player 0 plays a, and player 1 plays b

    @return:
    gradient - gradient[i,a,b]= derivative of loss w.r.t. reward[i,a,b]
    loss - half of the mean-squared strategy mismatch.
        Mean = average across both players.
        Strategy mismatch = difference between the expected reward that
        the player earns by cooperating (move 1) minus the expected reward that
        they earn by defecting (move 0).
    """
    logit = np.asarray(logit, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)

    exp_logit = np.exp(logit)

    pa = np.array([1 / (1 + exp_logit[0]), exp_logit[0] / (1 + exp_logit[0])], dtype=np.float64)
    pb = np.array([1 / (1 + exp_logit[1]), exp_logit[1] / (1 + exp_logit[1])], dtype=np.float64)

    da = np.array([-1, 1], dtype=np.float64)
    db = np.array([-1, 1], dtype=np.float64)

    loss = 1 / 2 * ((da.T @ reward[0] @ pb) ** 2 + (pa.T @ reward[1] @ db) ** 2)

    gradient = np.zeros_like(reward, dtype=np.float64)
    for a in range(2):
        for b in range(2):
            gradient[0, a, b] = (da.T @ reward[0] @ pb) * da[a] * pb[b]
            gradient[1, a, b] = (pa.T @ reward[1] @ db) * pa[a] * db[b]

    return gradient, loss


def mechanism_gradient_descent(logit, reward, nsteps, learningrate):
    """
    nsteps of gradient descent on the mean-squared strategy mismatch
    using simultaneous gradient ascent.

    @param:
    logit - The goal is to make this pair of strategies a Nash equlibrium:
        player i plays move 1 with probability 1/(1+exp(-logit[i])), else move 0.
    reward - Initial setting of the rewards.
        reward[i,a,b] is reward to player i if player 0 plays a, and player 1 plays b
    nsteps - number of steps of gradient descent to perform
    learningrate - learning rate

    @return:
    path - path[t,i,a,b] is the reward to player i of the moves (a,b) after t steps
      of gradient descent (path[0,:,:,:] = initial reward).
    loss - loss[t] is half of the mean-squared strategy mismatch at iteration [t].
        Mean = average across both players.
        Strategy mismatch = difference between the expected reward that
        the player earns by cooperating (move 1) minus the expected reward that
        they earn by defecting (move 0).
    """
    reward = np.asarray(reward).copy()

    path = np.zeros((nsteps, *reward.shape))
    all_losses = np.zeros((nsteps))

    path[0] = reward
    gradient, all_losses[0] = mechanism_gradient(logit, reward)

    curr_reward = reward.copy()
    for i in range(1, nsteps):
        gradient, loss = mechanism_gradient(logit, curr_reward)

        curr_reward = curr_reward - learningrate * gradient
        path[i] = curr_reward
        all_losses[i] = loss

    return path, all_losses
