"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""

import numpy as np

epsilon = 1e-3


def compute_transition(model):
    """
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    """
    M, N = model.R.shape
    P = np.zeros((M, N, 4, M, N))
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    for r in range(M):
        for c in range(N):
            if model.TS[r, c]:
                P[r, c, :, :, :] = 0
                continue
            for a in range(4):
                prob_stay = 1  # The probability that no action is executed
                for a_prime in range(3):
                    prob = model.D[r, c, a_prime]
                    match a_prime:
                        case 0:
                            direction = directions[a]
                        case 1:
                            direction = directions[(a - 1) % 4]
                        case 2:
                            direction = directions[(a + 1) % 4]
                    dr, dc = direction
                    new_r, new_c = r + dr, c + dc
                    if not (0 <= new_r < M and 0 <= new_c < N) or model.W[new_r, new_c]:
                        continue
                    prob_stay -= prob
                    P[r, c, a, new_r, new_c] += prob
                P[r, c, a, r, c] += prob_stay
    return P


def compute_utility(model, U_current, P):
    """
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    """
    M, N = model.R.shape
    U_next = np.zeros((M, N))

    # Update U using value iteration
    for r in range(M):
        for c in range(N):
            if model.TS[r, c]:
                U_next[r, c] = U_current[r, c]
                continue
            max_expected_utility = 0
            for a in range(4):
                expected_utility = 0
                for new_r in range(M):
                    for new_c in range(N):
                        prob = P[r, c, a, new_r, new_c]
                        expected_utility += prob * U_current[new_r, new_c]
                max_expected_utility = max(max_expected_utility, expected_utility)
            # Use the current utility to compute the new expected utility
            U_next[r, c] = model.R[r, c] + model.gamma * max_expected_utility

    return U_next


def value_iterate(model):
    """
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    """
    M, N = model.R.shape
    P = compute_transition(model)
    U_current = np.zeros((M, N))
    U_current[model.TS] = model.R[model.TS]

    # Update U until convergence
    while True:
        U_next = compute_utility(model, U_current, P)
        diff = np.abs(U_next - U_current)
        if np.max(diff) >= epsilon:
            U_current = U_next.copy()
            continue
        else:
            break

    return U_next
