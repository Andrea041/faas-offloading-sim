import os
import subprocess

from transfer_learning import TL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import yaml
import math
from collections import deque
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import transfer_learning

TRAIN = None
TRANSFER = True

MODEL_DIR = "dqn_results"
MODEL_PATH = os.path.join(MODEL_DIR, "model")
COMMIT_MSG = "Auto: aggiornato modello"

class DQN():
    def __init__(self, node_name, isStable, close_the_door_time):
        global TRAIN
        TRAIN = not isStable
        self.close_the_door_time = close_the_door_time
        node_found = False
        if TRANSFER:
            self.tl = TL(transfer_learning.load_tl_model())
        with open("dqn_config.yml", 'r') as file:
            config = yaml.safe_load(file)
            for node in config["nodes"]:
                if node["name"] == node_name:
                    self.state_size = node["state_size"]
                    self.action_size = node["action_size"]
                    # [(state, action, reward, next_state, next_allowed_actions)]
                    self.memory = []
                    self.memory = deque(maxlen=10000)
                    # dizionario {event:[state,action]} in attesa di:
                    #   - next_state & next_allowed_actions
                    #   - reward
                    self.pending_memory = {}
                    self.gamma = node["gamma"]
                    self.epsilon = node["epsilon"]
                    self.epsilon_min = node["epsilon_min"]
                    self.epsilon_decay = node["epsilon_decay"]
                    self.fraction_of_decay = node["fraction_of_decay"]
                    self.learning_rate = node["learning_rate"]
                    self.batch_size = node["batch_size"]
                    self.stable = isStable
                    self.stable_treshold = node["stable_treshold"]
                    self.train_round = 0
                    self.train_every = node["train_every"]
                    self.w1 = node["w1"]
                    self.w2 = node["w2"]
                    if not TRANSFER:
                        self.model = self._build_model()
                    else:
                        self.model = self.tl.build_tl_model()
                    node_found = True
                    break
        if not node_found:
            print("ERROR: node '" + node_name + "' not found in 'dqn_config.yml'!")
            exit(1)

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, action_filter):
        allowed_actions = [i for i, value in enumerate(action_filter) if value]
        if TRAIN and np.random.rand() <= self.epsilon:
            return random.choice(allowed_actions), True
        act_values = self.model.predict(state, verbose=0)[0].tolist()
        for val in sorted(act_values, reverse=True):
            if act_values.index(val) in allowed_actions:
                return act_values.index(val), False


    def learn(self, time):
        loss = []
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, next_allowed_actions in minibatch:
            target = reward
            if next_state is not None:
                act_values = self.model.predict(next_state, verbose=0)[0].tolist()
                for val in sorted(act_values, reverse=True):
                    if act_values.index(val) in next_allowed_actions:
                        target = reward + self.gamma * val
                        break
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            loss.append(history.history['loss'][0])
        if self.epsilon > self.epsilon_min:
            if self.fraction_of_decay > 0:
                self.epsilon = self.exponential_decay(time)
            else:
                self.epsilon *= self.epsilon_decay
        return loss


    def exponential_decay(self, now):
        alpha = -math.log(self.epsilon_min) / (self.close_the_door_time * self.fraction_of_decay)
        new_value = math.exp(-alpha * now)
        return new_value


    def save(self):
        if not TRANSFER:
            save_model(self.model, "dqn_results/model.keras")

        # Questo è per serverledge
        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32, name="state_input")])
        def inference_fn(state_input):
            return {"action": self.model(state_input)}

        tf.saved_model.save(
            self.model,
            "dqn_results/model",
            signatures=inference_fn)

        # Questo è per il transfer learning
        if TRANSFER:
            save_model(self.model, "dqn_results/model_tl.keras", include_optimizer=False)

        try:
            subprocess.run(["git", "config", "user.name", "Andrea041"], check=True)
            subprocess.run(["git", "config", "user.email", "aandreo.2001@gmail.com"], check=True)

            subprocess.run(["git", "add", MODEL_PATH], check=True)
            subprocess.run(["git", "commit", "-m", COMMIT_MSG], check=False)

            subprocess.run(["git", "push"], check=True)

        except subprocess.CalledProcessError as e:
            print(f"Push model error: {e}")


    def load(self):
        if not TRANSFER:
            self.model = load_model("dqn_results/model.keras")
        else:
            self.model = load_model("dqn_results/model_tl.keras")