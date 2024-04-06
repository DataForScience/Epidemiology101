### −∗− mode : python ; −∗−
# @file NetworkEpiModel.py
# @author Bruno Goncalves
######################################################

import networkx as nx
import numpy as np
from numpy import linalg
from numpy import random
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
from EpiModel import *
from collections import Counter


class NetworkEpiModel(EpiModel):
    def __init__(self, network, compartments=None):
        super(NetworkEpiModel, self).__init__(compartments)
        self.network = network
        self.kavg_ = 2*network.number_of_edges()/network.number_of_nodes() 
        self.spontaneous = {}
        self.interactions = {}

    def integrate(self, timesteps, **kwargs):
        raise NotImplementedError("Network Models don't support numerical integration")

    def add_interaction(self, source, target, agent, rate, rescale=False):
        if rescale:
            rate /= self.kavg_

        super(NetworkEpiModel, self).add_interaction(source, target, agent=agent, rate=rate)

        if source not in self.interactions:
            self.interactions[source] = {}

        if target not in self.interactions[source]:
            self.interactions[source] = {}

        self.interactions[source][agent] = {'target': target, 'rate': rate}
        
    def add_spontaneous(self, source, target, rate):
        super(NetworkEpiModel, self).add_spontaneous(source, target, rate=rate)
        if source not in self.spontaneous:
            self.spontaneous[source] = {}

        if target not in self.spontaneous[source]:
            self.spontaneous[source] = {}

        self.spontaneous[source][target] = rate


    def simulate(self, timesteps, seeds, **kwargs):
        """Stochastically simulate the epidemic model"""
        pos = {comp: i for i, comp in enumerate(self.transitions.nodes())}
        N = self.network.number_of_nodes()

        population = np.zeros((timesteps, N), dtype='str')

        comps = list(self.transitions.nodes)
        time = np.arange(1, timesteps, 1, dtype='int')

        susceptible = self._get_susceptible().pop()

        active_nodes = set()
        current_active = set()
        active_states = self._get_active()

        for node in range(N):
            if node in seeds:
                population[0, node] = seeds[node]
                active_nodes.add(node)
            else:
                population[0, node] = susceptible

        infections = self._get_infections()

        for t in time:
            population[t] = np.copy(population[t-1])

            if len(active_nodes) == 0:
                continue

            current_active = list(active_nodes)
            np.random.shuffle(current_active)

            for node_i in current_active:
                state_i = population[t-1, node_i]

                if state_i in infections:
                    # contact each neighbour to see if we infect them
                    NN = list(self.network.neighbors(node_i))
                    np.random.shuffle(NN)

                    for node_j in NN:
                        state_j = population[t-1, node_j]

                        if state_j in infections[state_i]:
                            prob = np.random.random()

                            if prob < infections[state_i][state_j]['rate']:
                                new_state = infections[state_i][state_j]['target']
                                population[t, node_j] = new_state

                                if new_state in active_states:
                                    active_nodes.add(node_j)
 
                if state_i in self.spontaneous:
                    n_trans = len(self.spontaneous[state_i])

                    prob = np.zeros(len(pos))

                    for target in self.spontaneous[state_i]:
                        prob[pos[target]] = self.spontaneous[state_i][target]

                    prob[pos[state_i]] = 1-np.sum(prob)

                    new_state = comps[np.argmax(random.multinomial(1, prob))]

                    if new_state != state_i:
                        population[t, node_i] = new_state
                        
                        active_nodes.add(node_i)

                        if new_state not in active_states:
                            active_nodes.remove(node_i)
                        
                        continue


        self.population_ = pd.DataFrame(population)
        self.values_ = pd.DataFrame.from_records(self.population_.apply(lambda x: Counter(x), axis=1)).fillna(0).astype('int')

    def R0(self):
        if 'R' not in set(self.transitions.nodes):
            return None
        return np.round(super(NetworkEpiModel, self).R0()*self.kavg_, 2)

if __name__ == '__main__':

    from tqdm import tqdm

    N = 100
    G = nx.erdos_renyi_graph(N, p=1.)

    SIR = NetworkEpiModel(G)
    SIR.add_interaction('S', 'I', 'I', 0.2)
    #SIR.add_interaction('S', 'E', 'Is', 0.2)
    #SIR.add_spontaneous('E', 'Ia', 0.5*0.1)
    #SIR.add_spontaneous('E', 'Is', 0.5*0.1)
    #SIR.add_spontaneous('Ia', 'R', 0.1)
    SIR.add_spontaneous('I', 'R', 0.1)

    print("kavg=", SIR.kavg_)
    print(SIR.transitions.edges(data=True))

    SIR._get_active()

    #print("R0 =", SIR.R0())

    fig, ax = plt.subplots(1)

    values = []
    Nruns = 1000

    for i in tqdm(range(Nruns), total=Nruns):
        SIR.simulate(100, seeds={30: 'I', 60:'I', 90:'I'})
        ax.plot(SIR.I/N, lw=.1, c='b')
        if SIR.R.max() > 10:
            values.append(SIR.I)

    ax.set_xlabel('Time')
    ax.set_ylabel('Population')

    values =  pd.DataFrame(values).T
    values.columns = np.arange(values.shape[1])
    ax.plot(values.mean(axis=1)/N, lw=2, c='r')
    ax.plot(values.median(axis=1)/N, lw=2, c='r', linestyle=':')

    SIR = EpiModel()
    SIR.add_interaction('S', 'I', 'I', 0.2)
    SIR.add_spontaneous('I', 'R', 0.1)
    SIR.integrate(100, S=N-3, I=3, R=0)
    ax.plot(SIR.I/N, lw=2, c='c')
    fig.savefig('SIR.png')