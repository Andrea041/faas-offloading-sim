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
    def __init__(self, node_region):
        # TODO: passare tutti questi valori da un file di configurazione
        self.state_size = 13
        self.action_size = 4
        self.epsilon = 1.0  # grado di esplorazione iniziale
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def learn(self, state, action, reward):
        np_state = np.array(state)
        np_state = np_state.reshape((1, len(state)))
        target_f = self.model.predict(np_state, verbose=0)
        target_f[0][action] = reward
        self.model.fit(np_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



class DQN(Policy):
    '''
    Stati:
    - can_execute_locally (boolean)
    - can_execute_on_edge (boolean)
    - can_execute_on_cloud (boolean)
    - local_cold_start (boolean)
    - function_id (int)
    - class_id (int)
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

        self.agent = Agent(node.region.name)

        self.how_many_offload_allowed = 5

        self.stats = {
            "SchedulerDecision.EXEC": [0],
            "SchedulerDecision.OFFLOAD_EDGE": [0],
            "SchedulerDecision.OFFLOAD_CLOUD": [0],
            "SchedulerDecision.DROP": [0],
            "SchedulerDecision.FORCED_DROP": [0],

            "right_drop_local": [0,0,0],
            "right_drop_edge": [0,0,0],
            "right_drop_both": [0,0,0],
            
            "reward": [0],

            "curr_memory": [0],

            "critical" : [0,0,0,0,0],
            "best-effort" : [0,0,0,0,0],
            "deferrable" : [0,0,0,0,0],

            "critical_offload_chain" : [0,0,0,0,0,0],
            "best-effort_offload_chain" : [0,0,0,0,0,0],
            "deferrable_offload_chain" : [0,0,0,0,0,0]
        }

        self.time = 0


    def schedule(self, e):

        self.stats["curr_memory"].append(e.node.curr_memory)

        if not SHOW_PRINTS and self.simulation.t - self.time > 0:
            print("[{:.2f}]".format(self.simulation.t))
            self.time += 10

        f = e.function
        c = e.qos_class

        # recupero lo stato ed un eventuale miglior nodo edge su cui fare l'offload e scelgo l'azione
        state, best_edge_node = self.get_state(f, c, e)
        np_state = np.array(state)
        np_state = np_state.reshape((1, len(state)))
        action = self.possible_decisions[self.agent.act(np_state)]

        arrival_time = e.original_arrival_time if e.original_arrival_time is not None else self.simulation.t
        
        # controllo che l'azione scelta sia ammessa
        can_execute_on_edge = state[1] and (len(e.offloaded_from) < self.how_many_offload_allowed)
        can_execute_on_cloud = state[2] and (len(e.offloaded_from) < self.how_many_offload_allowed)
        # 'can_execute_locally' non lo prendo dallo stato perchè prima era senza 'reclaim_memory'
        # in questo caso verrà eseguito il reclaim solo se la decisione è effettivamente EXEC
        if action == SchedulerDecision.EXEC and not self.can_execute_locally(f) or \
            action == SchedulerDecision.OFFLOAD_CLOUD and not can_execute_on_cloud or \
            action == SchedulerDecision.OFFLOAD_EDGE and not can_execute_on_edge:
                # statistiche separata per 'drop forzato'
                self.stats["SchedulerDecision.FORCED_DROP"].append(arrival_time)
                self.stats[c.name][self.possible_decisions.index(SchedulerDecision.DROP)+1] += 1
                self.stats[c.name+"_offload_chain"][len(e.offloaded_from)] += 1
                return (SchedulerDecision.DROP, action, state, None)

        # serve perchè best_edge_node viene ritornato come 'target_node' anche se la decisione è cloud
        # e quindi verrebbe selezionato il nodo edge invece di uno cloud al ritorno
        if action != SchedulerDecision.OFFLOAD_EDGE:
            best_edge_node = None

        # se faccio l'offload aggiungo l'evento tra quelli pending in attesa del reward
        if action == SchedulerDecision.OFFLOAD_CLOUD or action == SchedulerDecision.OFFLOAD_EDGE:
            self.pending_events[self.simulation.t] = [state,action,e]
            if SHOW_PRINTS and action in (SchedulerDecision.OFFLOAD_CLOUD, SchedulerDecision.OFFLOAD_EDGE):
                print("[{:.2f}]".format(self.simulation.t), e.node, end=" ")
                print("offloaded to cloud" if action == SchedulerDecision.OFFLOAD_CLOUD else f"offloaded to {best_edge_node}")

        self.stats[str(action)].append(arrival_time)
        self.stats[c.name][self.possible_decisions.index(action)] += 1
        if action == SchedulerDecision.EXEC or action == SchedulerDecision.DROP:
            self.stats[c.name+"_offload_chain"][len(e.offloaded_from)] += 1

        if action == SchedulerDecision.DROP and not state[0] and not can_execute_on_edge:
            if c.name == "critical":
                self.stats["right_drop_both"][0] += 1
            elif c.name == "best-effort":
                self.stats["right_drop_both"][1] += 1
            else:
                self.stats["right_drop_both"][2] += 1
        if action == SchedulerDecision.DROP and not state[0]:
            if c.name == "critical":
                self.stats["right_drop_local"][0] += 1
            elif c.name == "best-effort":
                self.stats["right_drop_local"][1] += 1
            else:
                self.stats["right_drop_local"][2] += 1
        if action == SchedulerDecision.DROP and not can_execute_on_edge:
            if c.name == "critical":
                self.stats["right_drop_edge"][0] += 1
            elif c.name == "best-effort":
                self.stats["right_drop_edge"][1] += 1
            else:
                self.stats["right_drop_edge"][2] += 1

        return(action, None, state, best_edge_node)


    def get_state(self, f, c, e):
        can_execute_locally = self.can_execute_locally(f, reclaim_memory=False)
        best_edge_node = self.get_best_edge_node(f.memory)
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
        return [can_execute_locally, can_execute_on_edge, can_execute_on_cloud, local_cold_start] + function_one_hot + class_one_hot + [has_been_offloaded, time_left], best_edge_node


    # Seleziona tra i nodi edge disponibili all'offload, uno random tra quelli con speedup maggiore 
    def get_best_edge_node(self, required_memory):
        peers = self._get_edge_peers()
        chosen = []
        curr_speedup = 0
        for peer in peers:
            if peer.curr_memory*peer.peer_exposed_memory_fraction >= required_memory:
                if peer.speedup > curr_speedup:
                    curr_speedup = peer.speedup
                    chosen = [peer]
                elif peer.speedup == curr_speedup:
                    chosen.append(peer)
        if len(chosen) < 1:
            return None
        return self.simulation.node_choice_rng.choice(chosen)


    def get_reward(self, state, action, original_decision, event, duration):
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
            else:
                reward = -c.deadline_penalty
            if SHOW_PRINTS:
                print("[{:.2f}]".format(self.simulation.t), event.node, end=" ")
                print("EXEC from {} : {}".format(event.offloaded_from[-1], reward) if bool(event.offloaded_from) else "EXEC : {}".format(reward))
        elif action == SchedulerDecision.DROP:
            reward = -c.drop_penalty
            # se il drop è avvenuto a seguito di un'azione non eseguibile, facciamo pagare la penalty per la decisione originale
            if original_decision is not None:
                action = original_decision
            # serve nelle risposte a ritroso dell'offload per far capire che è avvenuto il drop
            duration = -1
            if SHOW_PRINTS:
                print("[{:.2f}]".format(self.simulation.t), event.node, end=" ")
                print("DROP from {} : {}".format(event.offloaded_from[-1], reward) if bool(event.offloaded_from) else "DROP : {}".format(reward))
        elif action == SchedulerDecision.OFFLOAD_CLOUD or action == SchedulerDecision.OFFLOAD_EDGE:
            # se la durata è negativa è avvenuto un drop
            if duration < 0:
                reward = -c.drop_penalty
            elif c.max_rt <= 0.0 or duration <= c.max_rt:
                reward = c.utility
            else:
                reward = -c.deadline_penalty
            if SHOW_PRINTS:
                print("[{:.2f}]".format(self.simulation.t), event.node,"OFFLOAD completed :",reward)

        self.agent.learn(state,self.possible_decisions.index(action),reward)

        self.stats["reward"].append(reward)

        # propaga il reward a ritroso
        if bool(event.offloaded_from):
            n = event.offloaded_from[-1]
            node_policy = self.simulation.node2policy[n]
            pending_event = node_policy.pending_events[event.original_arrival_time]
            node_policy.get_reward(pending_event[0], pending_event[1], None, pending_event[2], duration)
