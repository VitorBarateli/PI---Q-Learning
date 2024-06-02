import random as rnd
import numpy as np
from pyfiglet import Figlet
import math
from grid import AbaloneGrid, Hex
import config as config
import algorithms as algs
from q_learning import Q_Learning

custom_fig = Figlet(font='big')

# Q-learning parâmetros
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Diretório salvar/carregar Q-table
Q_TABLE_FILE = "q_table.pkl"

# Inicializar Q-Learning
q_learning = Q_Learning(alpha=alpha, gamma=gamma, epsilon=epsilon)

# Carregar Q-table se existir
q_learning.q_table = q_learning.load_q_table(Q_TABLE_FILE)

# Inicializar o tabuleiro
initial_position = config.initialize('tabuleiro')
grid = AbaloneGrid(initial_position)
win = False

while not win:
    black_score = 0
    white_score = 0
    iterations = 0
    accum_node_count = 0
    node_count = 0
    depth = 3
    simulations = 1
    rnd.seed(4106)

    print("__________________________________________")
    print("\n White: ", white_score, "\t\tBlack: ", black_score)
    print("__________________________________________\n\n")
    print(grid.display)
    print("\n")

    ###### GAME START ######
    while True:
        ################## Black ####################
        # move
        curr_white = grid.query.marbles(grid.WHITE, True)
        state = grid.deep_copy(raw=True)
        possible_moves = list(grid.moves(grid.BLACK))
        action = q_learning.choose_action(state, q_learning.q_table, grid)

        try:
            action_index = possible_moves.index(action)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Ação: {action} não é um movimento possível")
            continue

        block, direction = possible_moves[action_index]
        grid.move(block, direction)

        # Calcular a recompensa
        reward = q_learning.calculate_reward(grid, grid.query.marbles(grid.BLACK, True), curr_white, action)

        next_state = grid.deep_copy(raw=True)
        q_learning.update_q_value(state, action, reward, next_state, q_learning.q_table, grid)

        print("__________________________________________")
        print("\n White: ", white_score, "\t\tBlack: ", black_score)
        print("__________________________________________\n")
        print("Iteração: ", iterations, "\n")
        print(grid.display)
        print("__________________________________________")
        print("\nBlack moveu: ", (block, direction))
        print("__________________________________________\n\n")

        # Verificar vitória
        if grid.query.check_win(grid.BLACK):
            print("Black ganhou!")
            win = True
            break

        # Atualizar pontuação
        black_score += curr_white - grid.query.marbles(grid.WHITE, True)
        curr_white = grid.query.marbles(grid.WHITE, True)

        
        ################## White ####################
        # move
        move = None
        curr_black = grid.query.marbles(grid.BLACK, True)
        _, move = algs.alphabeta(grid, depth, grid.WHITE, -math.inf, math.inf)
        accum_node_count += algs.node_count
        node_count = algs.node_count
        grid.move(move[0], move[1])

        print("__________________________________________")
        print("\n White: ", white_score, "\t\tBlack: ", black_score)
        print("__________________________________________\n")
        print("Iteração: ", iterations, "\n")
        print(grid.display)
        print("______________________________________")
        print("\nWhite move: ", move, sep="")
        print("______________________________________\n\n")
        
        node_count = 0
        algs.node_count = 0
        print("\n")

        # Verificar vitória
        if grid.query.check_win(grid.WHITE):
            print("White venceu!")
            print("______________________________________\n\n")
            win = True
            break

        # Atualizar pontuação
        white_score += curr_black - grid.query.marbles(grid.BLACK, True)
        curr_black = grid.query.marbles(grid.BLACK, True)

        iterations += 1



# Salvar Q-table
q_learning.save_q_table(q_learning.q_table, Q_TABLE_FILE)