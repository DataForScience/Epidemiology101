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
        self.spontaneous = {}
        self.interactions = {}

    def integrate(self, timesteps, **kwargs):
        raise NotImplementedError("Network Models don't support numerical integration")

    def add_interaction(self, source, target, agent, rate):        
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

        susceptible = self._get_susceptible()

        for node in range(N):
            if node in seeds:
                population[0, node] = seeds[node]
            else:
                population[0, node] = susceptible

        infectious = self._get_infectious()

        for t in time:
            for node_i in self.network.nodes():
                state_i = population[t-1, node_i]
                population[t, node_i] = state_i

                if state_i in self.spontaneous:
                    n_trans = len(self.spontaneous[state_i])

                    prob = np.zeros(len(pos))

                    for target in self.spontaneous[state_i]:
                        prob[pos[target]] = self.spontaneous[state_i][target]

                    prob[pos[state_i]] = 1-np.sum(prob)

                    new_state = comps[np.argmax(random.multinomial(1, prob))]

                    if new_state != state_i:
                        population[t, node_i] = new_state
                        continue

                if state_i in self.interactions:
                    for node_j in self.network.neighbors(node_i):
                        state_j = population[t-1, node_j]

                        if state_j in self.interactions[state_i]:
                            prob = np.random.random()

                            if prob < self.interactions[state_i][state_j]['rate']:
                                population[t, node_i] = self.interactions[state_i][state_j]['target']
                                break

                        population[t, node_i] = population[t-1, node_i]

        self.population_ = pd.DataFrame(population)
        self.values_ = pd.DataFrame.from_records(self.population_.apply(lambda x: Counter(x), axis=1)).fillna(0).astype('int')

if __name__ == '__main__':

    G = nx.erdos_renyi_graph(100, p=1.)

    SIR = NetworkEpiModel(G)
    SIR.add_interaction('S', 'I', 'I', 0.2/100)
    #SIR.add_interaction('S', 'E', 'Is', 0.2)
    #SIR.add_spontaneous('E', 'Ia', 0.5*0.1)
    #SIR.add_spontaneous('E', 'Is', 0.5*0.1)
    #SIR.add_spontaneous('Ia', 'R', 0.1)
    SIR.add_spontaneous('I', 'R', 0.1)

    print("R0 =", SIR.R0())

    N = 100
    fig, ax = plt.subplots(1)

    values = []
    Nruns = 100

    for i in range(Nruns):
        SIR.simulate(365, seeds={50:'I'})
        ax.plot(SIR.I/N, lw=.1, c='b')
        if SIR.I.max() > 10:
            values.append(SIR.I)

    ax.set_xlabel('Time')
    ax.set_ylabel('Population')

    values =  pd.DataFrame(values).T
    values.columns = np.arange(values.shape[1])
    ax.plot(values.median(axis=1)/N, lw=1, c='r')
    fig.savefig('SIR.png')