import numpy as np
import pandas as pd
import scipy.stats as stats
from policy import Policy, SchedulerDecision
from dqn import DQN

TRAIN = True
SHOW_PRINTS = False


class RL(Policy):
    '''
    Stati:
    - perc_available_local_memory (float)
    - can_execute_on_edge (boolean)
    - function_id (one_hot)
    - class_id (one_hot)
    - has_been_offloaded (boolean)
    '''
    def __init__(self, simulation, node, policy, close_the_door_time):
        super().__init__(simulation, node)

        cloud_region = node.region.default_cloud
        cloud_nodes = [n for n in self.simulation.infra.get_region_nodes(cloud_region) if n.total_memory>0]
        self.cloud = self.simulation.node_choice_rng.choice(cloud_nodes, 1)[0]

        # dizionario {original_arrival_time:[state,action,event]} utilizzato per eventi di offload in attesa del reward
        self.pending_events = {}

        self.possible_decisions = list(SchedulerDecision)

        if policy == "dqn":
            self.agent = DQN(node.name, not TRAIN, close_the_door_time)
        else:
            print("[{:.2f}]".format(self.t), "ERRORE: Unknown policy specified for RL!")
            exit(1)
        
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

            "epsilon": [0],

            "ExploreSchedulerDecision.EXEC": [0],
            "ExploreSchedulerDecision.OFFLOAD_CLOUD": [0],
            "ExploreSchedulerDecision.OFFLOAD_EDGE": [0],
            "ExploreSchedulerDecision.DROP": [0],

            "ExploitSchedulerDecision.EXEC": [0],
            "ExploitSchedulerDecision.OFFLOAD_CLOUD": [0],
            "ExploitSchedulerDecision.OFFLOAD_EDGE": [0],
            "ExploitSchedulerDecision.DROP": [0],
        }

        self.arrivi = {}

        self.time = 0


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
            self.stats["epsilon"].append(self.agent.epsilon)

        f = e.function
        c = e.qos_class

        # recupero lo stato ed un eventuale miglior nodo edge su cui fare l'offload
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

        explore = False

        if action_filter.count(True) == 1:
            # se c'è una sola azione azione possibile, prendi quella
            action = action_filter.index(True)
        else:
            # altrimenti sceglila tra quelle possibili
            action, explore = self.agent.act(np_state,action_filter)
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
        if explore:
            self.stats["Explore"+str(action)].append(arrival_time)
        else:
            self.stats["Exploit"+str(action)].append(arrival_time)

        self.arrivi[arrival_time] = [f.name, c.name]

        self.stats[c.name][self.possible_decisions.index(action)] += 1

        # aggiungo l'elemento {event:[state,action]} al 'pending_memory'
        self.agent.pending_memory[self.simulation.t] = [np_state,self.possible_decisions.index(action)]

        return(action, best_edge_node)


    def action_filter(self, state, event):
        actions = [True]*4
        available_memory = event.node.total_memory * state[0]
        can_execute_locally = True if event.function in event.node.warm_pool or available_memory >= event.function.memory else False
        can_execute_on_cloud = len(event.offloaded_from) < self.how_many_offload_allowed
        can_execute_on_edge = state[1] and (len(event.offloaded_from) < self.how_many_offload_allowed)
        if not can_execute_locally:
            actions[0] = False
        if not can_execute_on_cloud:
            actions[1] = False
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
        self.simulation.functions.sort()
        function_index = self.simulation.functions.index(f)
        function_one_hot = [0] * len(self.simulation.functions)
        function_one_hot[function_index] = 1
        self.simulation.classes.sort()
        class_index = self.simulation.classes.index(c)
        class_one_hot = [0] * len(self.simulation.classes)
        class_one_hot[class_index] = 1
        has_been_offloaded = bool(e.offloaded_from)
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
        w1 = self.agent.w1
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
            loss = self.agent.learn(self.simulation.t)
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