import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import pandas as pd
import yaml
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from policy import Policy, SchedulerDecision

TRAIN = True
SHOW_PRINTS = False


class Agent():
    def __init__(self, node_name):
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
                    self.stable = not TRAIN     # in questo modo quando non faccio training risulta stabile
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
        if prob_sum != 1:
            action_probs = action_probs / prob_sum
        return np.random.choice(self.action_size, p=action_probs)

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
            
            ratio = tf.exp(tf.math.log(new_probs) - tf.math.log(old_probs))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            value_loss = tf.reduce_mean((returns - self.value_model(states, training=True)) ** 2)
            total_loss = policy_loss + value_loss
        
        grads = tape.gradient(total_loss, self.policy_model.trainable_variables + self.value_model.trainable_variables)
        self.policy_model.optimizer.apply_gradients(zip(grads[:len(self.policy_model.trainable_variables)], self.policy_model.trainable_variables))
        self.value_model.optimizer.apply_gradients(zip(grads[len(self.policy_model.trainable_variables):], self.value_model.trainable_variables))
        
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
        save_model(self.model, "dqn_results/model.keras")
        print(" ---> RICORDATI DI SPOSTARE ANCHE IL MODELLO! <---")

    def load(self):
        self.model = load_model("dqn_results/model.keras")



class PPO(Policy):
    '''
    Stati:
    - perc_available_local_memory (float)
    - can_execute_on_edge (boolean)
    - can_execute_on_cloud (boolean)
    - function_id (one_hot)
    - class_id (one_hot)
    - has_been_offloaded (boolean)
    '''
    def __init__(self, simulation, node):
        super().__init__(simulation, node)

        cloud_region = node.region.default_cloud
        cloud_nodes = [n for n in self.simulation.infra.get_region_nodes(cloud_region) if n.total_memory>0]
        self.cloud = self.simulation.node_choice_rng.choice(cloud_nodes, 1)[0]

        # dizionario {original_arrival_time:[state,action,event]} utilizzato per eventi di offload in attesa del reward
        self.pending_events = {}

        self.possible_decisions = list(SchedulerDecision)

        self.agent = Agent(node.name)
        if not TRAIN:
            self.agent.load()

        self.how_many_offload_allowed = 3

        self.cost_normalization_factor = self.get_cost_normalization_factor()

        self.stats = {
            "SchedulerDecision.EXEC": [0],
            "SchedulerDecision.OFFLOAD_CLOUD": [0],
            "SchedulerDecision.OFFLOAD_EDGE": [0],
            "SchedulerDecision.DROP": [0],
            
            "reward": [0],

            "standard" : [0,0,0,0],
            "critical-1" : [0,0,0,0],
            "critical-2" : [0,0,0,0],
            "batch" : [0,0,0,0],

            "loss": [0],

            "EXEC": [0,0],
            "OFFLOAD_CLOUD": [0,0],
            "OFFLOAD_EDGE": [0,0],

            "standard_reward" : [0,0,0],
            "critical-1_reward" : [0,0,0],
            "critical-2_reward" : [0,0,0],
            "batch_reward" : [0,0,0],

            "reward_cost": [0],
            "cost": [0],
        }

        self.time = 0
        self.periodic_stats = 360


    def get_cost_normalization_factor(self):
        cost_per_function = []
        for f in self.simulation.functions:
            percentile = stats.gamma.ppf(0.9999, 1.0/f.serviceSCV, scale=f.serviceMean*f.serviceSCV/self.cloud.speedup)
            # prendo il 50% in più per stare più tranquillo
            higher_percentile = percentile * 1.5
            cost_per_function.append(higher_percentile * f.memory/1024 * self.cloud.cost)
        return max(cost_per_function)

    def schedule(self, e):

        if not SHOW_PRINTS and self.simulation.t - self.time > 0:
            print("[{:.2f}]".format(self.simulation.t))
            self.time += 10

        f = e.function
        c = e.qos_class

        # recupero lo stato ed un eventuale miglior nodo edge su cui fare l'offload e scelgo l'azione
        state, best_edge_node = self.get_state(e)
        np_state = np.array(state)
        np_state = np_state.reshape((1, len(state)))

        # lista di 4 booleani che indica le azioni possibili
        action_filter = self.action_filter(state,e)

        # lista contenente gli indici delle azioni ammesse
        allowed_actions = [i for i, value in enumerate(action_filter) if value]

        # imposta lo stato attuale come 'next_state' per il primo degli elementi di 'agent.pending_memory' e le 'next_allowed_actions'
        if len(self.agent.pending_memory) > 0:
            for key, value in self.agent.pending_memory.copy().items():
                if len(value) < 4:
                    value.append(np_state)
                    value.append(allowed_actions)
                    if len(value) == 5:
                        # sposto l'esperienza completa dalla 'pending_memory' alla 'memory'
                        val = self.agent.pending_memory.pop(key)
                        self.agent.memory.append(tuple(val))
                    break

        if action_filter.count(True) == 1:
            # se c'è una sola azione azione possibile, prendi quella
            action = action_filter.index(True)
        else:
            # altrimenti sceglila tra quelle possibili
            action = self.agent.act(np_state,action_filter)
        action = self.possible_decisions[action]
        
        # se la decisione è EXEC eseguo 'can_execute_locally' nel caso servisse la 'reclaim_memory'
        if action == SchedulerDecision.EXEC:
            self.can_execute_locally(f)

        # serve perchè best_edge_node viene ritornato come 'target_node' anche se la decisione è cloud
        # e quindi verrebbe selezionato il nodo edge invece di uno cloud al ritorno
        if action != SchedulerDecision.OFFLOAD_EDGE:
            best_edge_node = None

        # se faccio l'offload aggiungo l'evento tra quelli pending in attesa del reward
        if action == SchedulerDecision.OFFLOAD_CLOUD or action == SchedulerDecision.OFFLOAD_EDGE:
            event = e # faccio una copia perchè non so se a livello di statistiche cambiare 'original_arrival_time' comporta problemi
            event.original_arrival_time = self.simulation.t
            self.pending_events[self.simulation.t] = [state,action,event]
            if SHOW_PRINTS and action in (SchedulerDecision.OFFLOAD_CLOUD, SchedulerDecision.OFFLOAD_EDGE):
                print("[{:.2f}]".format(self.simulation.t), e.node, end=" ")
                print("offloaded to cloud" if action == SchedulerDecision.OFFLOAD_CLOUD else f"offloaded to {best_edge_node}")

        arrival_time = e.original_arrival_time if e.original_arrival_time is not None else self.simulation.t
        self.stats[str(action)].append(arrival_time)
        self.stats[c.name][self.possible_decisions.index(action)] += 1

        # aggiungo l'elemento {event:[state,action]} al 'pending_memory'
        self.agent.pending_memory[self.simulation.t] = [np_state,self.possible_decisions.index(action)]

        return(action, best_edge_node)


    def action_filter(self, state, event):
        actions = [True]*4
        available_memory = event.node.total_memory * state[0]
        can_execute_locally = True if event.function in event.node.warm_pool or available_memory >= event.function.memory else False
        can_execute_on_edge = state[1] and (len(event.offloaded_from) < self.how_many_offload_allowed)
        #can_execute_on_cloud = state[2] and (len(event.offloaded_from) < self.how_many_offload_allowed)
        if not can_execute_locally:
            actions[0] = False
        # if not can_execute_on_cloud:
        #     actions[1] = False
        if not can_execute_on_edge:
            actions[2] = False
        return actions


    def get_state(self, e):
        f = e.function
        c = e.qos_class
        available_local_memory = e.node.curr_memory + sum([entry[0].memory for entry in e.node.warm_pool.pool])
        perc_av_loc_mem = available_local_memory / e.node.total_memory
        best_edge_node = self.get_best_edge_node(f.memory, e.offloaded_from)
        can_execute_on_edge = True if best_edge_node is not None else False
        # ASSUMO CHE CI SIA SEMPRE ALMENO 1 NODO CLOUD E CHE IL CLOUD ABBIA RISORSE DISPONIBILI
        can_execute_on_cloud = self.simulation.stats.cost / self.simulation.t * 3600 < self.budget
        function_index = self.simulation.functions.index(f)
        function_one_hot = [0] * len(self.simulation.functions)
        function_one_hot[function_index] = 1
        class_index = self.simulation.classes.index(c)
        class_one_hot = [0] * len(self.simulation.classes)
        class_one_hot[class_index] = 1
        has_been_offloaded = bool(e.offloaded_from)
        #return [perc_av_loc_mem, can_execute_on_edge, can_execute_on_cloud] + function_one_hot + class_one_hot + [has_been_offloaded], best_edge_node
        return [perc_av_loc_mem, can_execute_on_edge] + function_one_hot + class_one_hot + [has_been_offloaded], best_edge_node


    # Seleziona tra i nodi edge disponibili all'offload, uno random tra quelli con speedup maggiore 
    def get_best_edge_node(self, required_memory, offloaded_from):
        peers = self._get_edge_peers()
        chosen = []
        curr_speedup = 0
        for peer in peers:
            if peer not in offloaded_from and peer.curr_memory*peer.peer_exposed_memory_fraction >= required_memory:
                if peer.speedup > curr_speedup:
                    curr_speedup = peer.speedup
                    chosen = [peer]
                elif peer.speedup == curr_speedup:
                    chosen.append(peer)
        if len(chosen) < 1:
            return None
        return self.simulation.node_choice_rng.choice(chosen)


    def get_reward(self, action, event, duration, cost):
        c = event.qos_class

        if action == SchedulerDecision.EXEC:
            # Account for the time needed to send back the result
            if event.offloaded_from != None:
                curr_node = event.node
                for remote_node in reversed(event.offloaded_from):
                    duration += self.simulation.infra.get_latency(curr_node, remote_node)
                    curr_node = remote_node

            if c.max_rt <= 0.0 or duration <= c.max_rt:
                reward = c.utility
                self.stats["EXEC"][0] += 1
                self.stats[c.name+"_reward"][0] += 1
            else:
                reward = -c.deadline_penalty
                self.stats["EXEC"][1] += 1
                self.stats[c.name+"_reward"][1] += 1
            if SHOW_PRINTS:
                print("[{:.2f}]".format(self.simulation.t), event.node, end=" ")
                print("EXEC from {} : {}".format(event.offloaded_from[-1], reward) if bool(event.offloaded_from) else "EXEC : {}".format(reward))
        elif action == SchedulerDecision.DROP:
            reward = -c.drop_penalty
            # serve nelle risposte a ritroso dell'offload per far capire che è avvenuto il drop
            duration = -1
            self.stats[c.name+"_reward"][2] += 1
            if SHOW_PRINTS:
                print("[{:.2f}]".format(self.simulation.t), event.node, end=" ")
                print("DROP from {} : {}".format(event.offloaded_from[-1], reward) if bool(event.offloaded_from) else "DROP : {}".format(reward))
        elif action == SchedulerDecision.OFFLOAD_CLOUD or action == SchedulerDecision.OFFLOAD_EDGE:
            # se la durata è negativa è avvenuto un drop
            if duration < 0:
                # non ci dovrebbe più entrare perchè gli offload vengono fatti solo se eseguibili
                print("[{:.2f}]".format(self.t), "ERRORE: An offload has been dropped!")
                exit(1)
            elif c.max_rt <= 0.0 or duration <= c.max_rt:
                reward = c.utility
                self.stats[c.name+"_reward"][0] += 1
                self.stats[str(action).split(".")[1]][0] += 1
            else:
                reward = -c.deadline_penalty
                self.stats[c.name+"_reward"][1] += 1
                self.stats[str(action).split(".")[1]][1] += 1
            if SHOW_PRINTS:
                print("[{:.2f}]".format(self.simulation.t), event.node,"OFFLOAD completed :",reward)

        self.stats["reward"].append(reward)
        
        self.stats["cost"].append(cost)

        # normalizzo il costo
        nomalized_cost = cost / self.cost_normalization_factor

        # pesi rispettivamente per utility/penalty e cost
        w1 = 0.5
        w2 = 1 - w1

        reward = w1 * reward - w2 * nomalized_cost

        self.stats["reward_cost"].append(reward)

        # aggiungo il reward nel 'pending_memory'
        arrival_time = event.original_arrival_time if event.original_arrival_time is not None else self.simulation.t
        if len(self.agent.pending_memory[arrival_time]) == 2:
            # se ancora non ci sono 'next_state' & 'next_allowed_actions'
            self.agent.pending_memory[arrival_time].append(reward)
        else:
            # se già ci sono 'next_state' & 'next_allowed_actions'
            self.agent.pending_memory[arrival_time].insert(2,reward)
            # sposto l'esperienza completa dalla 'pending_memory' alla 'memory'
            val = self.agent.pending_memory.pop(arrival_time)
            self.agent.memory.append(tuple(val))


    def train(self):        
        if TRAIN and len(self.agent.memory) >= self.agent.batch_size and (not self.agent.stable or self.agent.train_round%self.agent.train_every == 0):
            loss = self.agent.learn()
            self.stats["loss"].append(np.mean(loss))

        if self.agent.stable:
            self.agent.train_round += 1
            self.agent.train_round = self.agent.train_round%self.agent.train_every
        else:
            loss_list = self.stats["loss"]
            mean_loss = pd.Series(loss_list).rolling(window=len(loss_list), min_periods=1).mean().tolist()
            last_100_mean_loss = mean_loss[(len(mean_loss)-100):]
            delta = max(last_100_mean_loss) - min(last_100_mean_loss)
            if len(loss_list) > 100 and delta < self.agent.stable_treshold:
                self.agent.stable = True