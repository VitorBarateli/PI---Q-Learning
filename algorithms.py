import math

node_count = 0

def alphabeta(board, depth, maximizer, alpha, beta):
    if board.query.check_win(maximizer):
        return math.inf if maximizer else -math.inf, -1
    elif depth == 0:
        return heuristic(board), -1

    if maximizer:
        score = -math.inf

        def should_replace(x):
            return x > score
    else:
        score = math.inf

        def should_replace(x):
            return x < score

    move = -1

    successors = list(board.moves(maximizer))

    for successor in successors:
        global node_count
        node_count = node_count + 1

        action = successor
        state = board.deep_copy()
        state.move(*action)

        temp = alphabeta(state, depth - 1, not maximizer, alpha, beta)[0]

        if should_replace(temp):
            score = temp
            move = action
        if maximizer:
            alpha = max(alpha, temp)
        else:
            beta = min(beta, temp)
        if alpha >= beta:
            break

    return score, move


####################################### HEURISTIC ##########################################
def heuristic(board):
    center_proximity = board.query.center_proximity(False) - board.query.center_proximity(True)

    # cohesion
    cohesion = 0
    if abs(center_proximity) > 2:
        cohesion = len(list(board.query.populations(False))) - len(list(board.query.populations(True)))

    # number of marbles
    marbles = 0
    if abs(center_proximity) < 1.8:
        marbles = board.query.marbles(True, True) * 100 - board.query.marbles(False, True) * 100

    return center_proximity + cohesion + marbles
