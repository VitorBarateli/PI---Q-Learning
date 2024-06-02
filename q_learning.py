import random as rnd
import numpy as np
import pickle
from grid import Hex

class Q_Learning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    # Função salvar Q-table
    @staticmethod
    def save_q_table(q_table, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(q_table, f)

    # Função carregar Q-table
    @staticmethod
    def load_q_table(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    # Funçao para converter o estado para tipo hashable (tuple)
    @staticmethod
    def state_to_tuple(state):
        if isinstance(state, (list, tuple)):
            return tuple(map(Q_Learning.state_to_tuple, state))
        elif isinstance(state, dict):
            return tuple((k, Q_Learning.state_to_tuple(v)) for k, v in sorted(state.items()))
        else:
            return state

    # Escolher ação usando politica epsilon-greedy
    def choose_action(self, state, q_table, grid):
        state_tuple = self.state_to_tuple(state)

        if state_tuple not in q_table:
            possible_moves = list(grid.moves(grid.BLACK))
            q_table[state_tuple] = {move: 0.0 for move in possible_moves}

        if np.random.uniform(0, 1) < self.epsilon:
            action = rnd.choice(list(q_table[state_tuple].keys()))
        else:
            action = max(q_table[state_tuple], key=q_table[state_tuple].get)

        return action

    # Função calcular distância do centro do tabuleiro
    @staticmethod
    def distance_to_center(hex_position):
        center = Hex(0, 0)
        return abs(hex_position.x - center.x) + abs(hex_position.z - center.z)

    # Calcular a recompensa baseado na mudança de estado
    def calculate_reward(self, grid, previous_black, previous_white, action):
        reward = -0.04  # Recompensa padrão por fazer um movimento

        # Determinar se o movimento foi em direção ao centro
        block, direction = action
        new_positions = [Hex(m.x + direction[0], m.z + direction[1]) for m in block]
        avg_distance_before = sum(self.distance_to_center(m) for m in block) / len(block)
        avg_distance_after = sum(self.distance_to_center(m) for m in new_positions) / len(new_positions)

        if avg_distance_after < avg_distance_before:
            reward = -0.02  # Recompensa por mover em direção ao centro

        current_black = grid.query.marbles(grid.BLACK, True)
        current_white = grid.query.marbles(grid.WHITE, True)

        # Verificar se eliminou bola inimiga
        if previous_white > current_white:
            reward += 0.5 * (previous_white - current_white)

        # Verificar se perdeu uma bola
        if previous_black > current_black:
            reward -= 0.5 * (previous_black - current_black)

        # Verificar vitória/derrota
        if grid.query.check_win(grid.BLACK):
            reward += 1.0
        elif grid.query.check_win(grid.WHITE):
            reward -= 1.0

        return reward

    # Atualizar Q-value
    def update_q_value(self, state, action, reward, next_state, q_table, grid):
        state_tuple = self.state_to_tuple(state)
        next_state_tuple = self.state_to_tuple(next_state)

        if next_state_tuple not in q_table:
            possible_moves = list(grid.moves(grid.BLACK))
            q_table[next_state_tuple] = {move: 0.0 for move in possible_moves}

        best_next_action = max(q_table[next_state_tuple], key=q_table[next_state_tuple].get)
        q_table[state_tuple][action] = q_table[state_tuple][action] + self.alpha * (
                    reward + self.gamma * q_table[next_state_tuple][best_next_action] - q_table[state_tuple][action])