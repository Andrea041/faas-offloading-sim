import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from policy import Policy, SchedulerDecision

SHOW_PRINTS = False


class Agent():
    def __init__(self):
        # TODO: passare tutti questi valori da un file di configurazione
        self.state_size = 11
        self.action_size = 4
        # [(state, action, reward, next_state, next_allowed_actions)]
        self.memory = []
        # dizionario {event:[state,action]} in attesa di:
        #   - next_state & next_allowed_actions
        #   - reward
        self.pending_memory = {}
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 3
        self.model = self._build_model()

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
        if np.random.rand() <= self.epsilon:
            return random.choice(allowed_actions)
        act_values = self.model.predict(state, verbose=0)[0].tolist()
        for val in sorted(act_values, reverse=True):
            if act_values.index(val) in allowed_actions:
                return act_values.index(val)

    def learn(self):
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
            self.epsilon *= self.epsilon_decay
        return loss



class DQN(Policy):
    '''
    Stati:
    - perc_available_local_memory (float)
    - can_execute_on_edge (boolean)
    - can_execute_on_cloud (boolean)
    - local_cold_start (boolean)
    - function_id (one_hot)
    - class_id (one_hot)
    - has_been_offloaded (boolean)
    - time_left (float)
    '''
    def __init__(self, simulation, node):
        super().__init__(simulation, node)

        cloud_region = node.region.default_cloud
        cloud_nodes = [n for n in self.simulation.infra.get_region_nodes(cloud_region) if n.total_memory>0]
        self.cloud = self.simulation.node_choice_rng.choice(cloud_nodes, 1)[0]

        # dizionario {original_arrival_time:[state,action,event]} utilizzato per eventi di offload in attesa del reward
        self.pending_events = {}

        self.possible_decisions = list(SchedulerDecision)

        self.agent = Agent()

        self.how_many_offload_allowed = 3

        self.stats = {
            "SchedulerDecision.EXEC": [0],
            "SchedulerDecision.OFFLOAD_CLOUD": [0],
            "SchedulerDecision.OFFLOAD_EDGE": [0],
            "SchedulerDecision.DROP": [0],
            
            "reward": [0],

            "critical" : [0,0,0,0],
            "best-effort" : [0,0,0,0],
            "deferrable" : [0,0,0,0],

            "loss": [0],

            "EXEC": [0,0],
            "OFFLOAD_CLOUD": [0,0,0],
            "OFFLOAD_EDGE": [0,0,0],

            "critical_reward" : [0,0,0],
            "best-effort_reward" : [0,0,0],
            "deferrable_reward" : [0,0,0],
        }

        self.time = 0


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
        can_execute_on_cloud = state[2] and (len(event.offloaded_from) < self.how_many_offload_allowed)
        if not can_execute_locally:
            actions[0] = False
        if not can_execute_on_cloud:
            actions[1] = False
        if not can_execute_on_edge:
            actions[2] = False
        # se c'è almeno 1 soluzione, disabilita il drop
        if any(actions[:3]):
            actions[3] = False
        return actions


    def get_state(self, e):
        f = e.function
        c = e.qos_class
        available_local_memory = e.node.curr_memory + e.node.warm_pool.reclaim_memory(f.memory, reclaim_memory=False)
        perc_av_loc_mem = available_local_memory / e.node.total_memory
        best_edge_node = self.get_best_edge_node(f.memory, e.offloaded_from)
        can_execute_on_edge = True if best_edge_node is not None else False
        # ASSUMO CHE CI SIA SEMPRE ALMENO 1 NODO CLOUD E CHE IL CLOUD ABBIA RISORSE DISPONIBILI
        can_execute_on_cloud = self.simulation.stats.cost / self.simulation.t * 3600 < self.budget
        local_cold_start = not f in e.node.warm_pool
        function_index = self.simulation.functions.index(f)
        function_one_hot = [0] * len(self.simulation.functions)
        function_one_hot[function_index] = 1
        class_index = self.simulation.classes.index(c)
        class_one_hot = [0] * len(self.simulation.classes)
        class_one_hot[class_index] = 1
        has_been_offloaded = bool(e.offloaded_from)
        arrival_time = e.original_arrival_time if e.original_arrival_time is not None else self.simulation.t
        time_left = self.simulation.t - arrival_time - c.max_rt
        #return [perc_av_loc_mem, can_execute_on_edge, can_execute_on_cloud, local_cold_start] + function_one_hot + class_one_hot + [has_been_offloaded, time_left], best_edge_node
        # return [perc_av_loc_mem, can_execute_on_edge, can_execute_on_cloud, local_cold_start] + function_one_hot + class_one_hot + [has_been_offloaded], best_edge_node
        return [perc_av_loc_mem, can_execute_on_edge, can_execute_on_cloud] + function_one_hot + class_one_hot + [has_been_offloaded], best_edge_node


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


    def get_reward(self, action, event, duration):
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
                reward = -c.drop_penalty
                self.stats[c.name+"_reward"][2] += 1
                if action == SchedulerDecision.OFFLOAD_CLOUD:
                    self.stats["OFFLOAD_CLOUD"][2] += 1
                else:
                    self.stats["OFFLOAD_EDGE"][2] += 1
            elif c.max_rt <= 0.0 or duration <= c.max_rt:
                reward = c.utility
                self.stats[c.name+"_reward"][0] += 1
                if action == SchedulerDecision.OFFLOAD_CLOUD:
                    self.stats["OFFLOAD_CLOUD"][0] += 1
                else:
                    self.stats["OFFLOAD_EDGE"][0] += 1
            else:
                reward = -c.deadline_penalty
                self.stats[c.name+"_reward"][1] += 1
                if action == SchedulerDecision.OFFLOAD_CLOUD:
                    self.stats["OFFLOAD_CLOUD"][1] += 1
                else:
                    self.stats["OFFLOAD_EDGE"][1] += 1
            if SHOW_PRINTS:
                print("[{:.2f}]".format(self.simulation.t), event.node,"OFFLOAD completed :",reward)

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

        self.stats["reward"].append(reward)


    def train(self):
        if len(self.agent.memory) >= self.agent.batch_size:
            loss = self.agent.learn()
            self.stats["loss"].append(np.mean(loss))