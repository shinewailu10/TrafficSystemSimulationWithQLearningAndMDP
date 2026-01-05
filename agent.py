import numpy as np
import pickle 

class QLearner:
    def __init__(self, learning_rate=0.01, discount_factor=0.9, exploration_rate=1.0):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.actions = [0, 1, 2, 3]
        self.last_action = 0 

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, current_green):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)

        q_values = []
        for a in self.actions:
            q = self.get_q_value(state, a)
            if a == current_green:
                q += 0.05  
            q_values.append(q)

        return self.actions[np.argmax(q_values)]


    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        
        #Bellman Equation
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

    def save_model(self, filename="traffic_brain.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Brain saved to {filename}")

    def load_model(self, filename="traffic_brain.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Brain loaded from {filename}")
            return True
        except FileNotFoundError:
            print("No saved brain found. Starting fresh.")
            return False