import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import yaml
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class PPO():
    def __init__(self, node_name, isStable):
        node_found = False
        with open("dqn_config.yml", 'r') as file:
            config = yaml.safe_load(file)
            for node in config["nodes"]:
                if node["name"] == node_name:
                    self.state_size = node["state_size"]
                    self.action_size = node["action_size"]
                    # [(state, action, reward, next_state, next_allowed_actions)]
                    self.memory = []
                    # dizionario {event:[state,action]} in attesa di:
                    #   - next_state & next_allowed_actions
                    #   - reward
                    self.pending_memory = {}
                    self.gamma = node["gamma"]
                    self.learning_rate = node["learning_rate"]
                    self.batch_size = node["batch_size"]
                    self.stable = isStable
                    self.stable_treshold = node["stable_treshold"]
                    self.train_round = 0
                    self.train_every = node["train_every"]
                    
                    self.clip_ratio = 0.2
                    self.epochs = 10
                    
                    self.policy_model, self.value_model = self._build_model()
                    self.policy_optimizer = Adam(learning_rate=self.learning_rate)
                    self.value_optimizer = Adam(learning_rate=self.learning_rate)
                    
                    node_found = True
                    break
        if not node_found:
            print("ERROR: node '" + node_name + "' not found in 'dqn_config.yml'!")
            exit(1)

    def _build_model(self):
        policy_model = Sequential()
        policy_model.add(Input(shape=(self.state_size,)))
        policy_model.add(Dense(24, activation='relu'))
        policy_model.add(Dense(24, activation='relu'))
        policy_model.add(Dense(self.action_size, activation='softmax'))
        
        value_model = Sequential()
        value_model.add(Input(shape=(self.state_size,)))
        value_model.add(Dense(24, activation='relu'))
        value_model.add(Dense(24, activation='relu'))
        value_model.add(Dense(1, activation='linear'))
        
        policy_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        value_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        
        return policy_model, value_model

    def act(self, state, action_filter):
        action_probs = self.policy_model.predict(state, verbose=0)[0]
        action_probs = action_probs * np.array(action_filter).astype(int)
        prob_sum = np.sum(action_probs)
        if prob_sum == 0 or np.isnan(prob_sum):
            print("0 or NaN")
            return 3    # DROP
        elif prob_sum != 1:
            action_probs = action_probs / prob_sum
        return np.random.choice(self.action_size, p=action_probs)

    # def learn(self):
    #     minibatch = random.sample(self.memory, self.batch_size)

    #     states = np.squeeze(np.array([item[0] for item in minibatch]), axis=1)
    #     actions = np.array([item[1] for item in minibatch])
    #     rewards = np.array([item[2] for item in minibatch])
    #     next_states = np.squeeze(np.array([item[3] for item in minibatch]), axis=1)

    #     advantages, returns = self.get_advantages(rewards, next_states)
        
    #     actions_one_hot = tf.keras.utils.to_categorical(actions, self.action_size)
        
    #     with tf.GradientTape() as tape:
    #         old_probs = self.policy_model.predict(states, verbose=0)
    #         new_probs = self.policy_model(states, training=True)
            
    #         old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
    #         new_probs = tf.convert_to_tensor(new_probs, dtype=tf.float32)
            
    #         old_probs = tf.reduce_sum(actions_one_hot * old_probs, axis=1)
    #         new_probs = tf.reduce_sum(actions_one_hot * new_probs, axis=1)
            
    #         ratio = tf.exp(tf.math.log(new_probs) - tf.math.log(old_probs))
    #         clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
    #         surrogate1 = ratio * advantages
    #         surrogate2 = clipped_ratio * advantages
            
    #         policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
    #         value_loss = tf.reduce_mean((returns - self.value_model(states, training=True)) ** 2)
    #         total_loss = policy_loss + value_loss
        
    #     grads = tape.gradient(total_loss, self.policy_model.trainable_variables + self.value_model.trainable_variables)
    #     self.policy_model.optimizer.apply_gradients(zip(grads[:len(self.policy_model.trainable_variables)], self.policy_model.trainable_variables))
    #     self.value_model.optimizer.apply_gradients(zip(grads[len(self.policy_model.trainable_variables):], self.value_model.trainable_variables))
        
    #     return total_loss.numpy()

    def learn(self):
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.squeeze(np.array([item[0] for item in minibatch]), axis=1)
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.squeeze(np.array([item[3] for item in minibatch]), axis=1)

        advantages, returns = self.get_advantages(rewards, next_states)
        
        actions_one_hot = tf.keras.utils.to_categorical(actions, self.action_size)
        
        with tf.GradientTape() as tape:
            old_probs = self.policy_model.predict(states, verbose=0)
            new_probs = self.policy_model(states, training=True)
            
            old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
            new_probs = tf.convert_to_tensor(new_probs, dtype=tf.float32)
            
            old_probs = tf.reduce_sum(actions_one_hot * old_probs, axis=1)
            new_probs = tf.reduce_sum(actions_one_hot * new_probs, axis=1)
            
            # Verifica i NaN nelle probabilità
            tf.debugging.check_numerics(old_probs, message='Old_probs NaN Found')
            tf.debugging.check_numerics(new_probs, message='New_probs NaN Found')
            
            ratio = tf.exp(tf.math.log(new_probs + 1e-10) - tf.math.log(old_probs + 1e-10))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            value_loss = tf.reduce_mean((returns - self.value_model(states, training=True)) ** 2)
            total_loss = policy_loss + value_loss
            
            # Verifica i NaN nelle perdite
            tf.debugging.check_numerics(policy_loss, message='Policy_loss NaN Found')
            tf.debugging.check_numerics(value_loss, message='Value_loss NaN Found')
            tf.debugging.check_numerics(total_loss, message='Total_loss NaN Found')
        
        grads = tape.gradient(total_loss, self.policy_model.trainable_variables + self.value_model.trainable_variables)
        
        # Verifica i NaN nei gradienti
        #tf.debugging.check_numerics(grads, message='Grads NaN Found')
        
        # Calcola la norma dei gradienti
        gradients_norm = tf.linalg.global_norm(grads)
        print(f'{self.stable} | Gradient Norm: {gradients_norm.numpy()}')

        # Controlla per gradienti esplosivi
        if gradients_norm > 100:  # Soglia di esempio, può essere adattata
            print("Warning: Exploding gradients detected!")

        # Gradient Clipping
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        self.policy_model.optimizer.apply_gradients(zip(grads[:len(self.policy_model.trainable_variables)], self.policy_model.trainable_variables))
        self.value_optimizer.apply_gradients(zip(grads[len(self.policy_model.trainable_variables):], self.value_model.trainable_variables))
        
        return total_loss.numpy()

    def get_advantages(self, rewards, next_states):
        values = self.value_model.predict(next_states, verbose=0)
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t]
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * last_gae_lam
        returns = advantages + values
        return advantages, returns


    def save(self):
        save_model(self.policy_model, "dqn_results/policy_model.keras")
        save_model(self.value_model, "dqn_results/value_model.keras")
        print(" ---> RICORDATI DI SPOSTARE ANCHE IL MODELLO! <---")

    def load(self):
        self.policy_model = load_model("dqn_results/policy_model.keras")
        self.value_model = load_model("dqn_results/value_model.keras")