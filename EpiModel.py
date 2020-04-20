### −∗− mode : python ; −∗−
# @file EpiModel.py
# @author Bruno Goncalves
######################################################

import networkx as nx
import numpy as np
from numpy import random
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt

class EpiModel(object):
    """Simple Epidemic Model Implementation
    
        Provides a way to implement and numerically integrate 
    """
    def __init__(self, compartments=None):
        self.transitions = nx.MultiDiGraph()
        
        if compartments is not None:
            self.transitions.add_nodes_from([comp for comp in compartments])
    
    def add_interaction(self, source, target, agent, rate):        
        self.transitions.add_edge(source, target, agent=agent, rate=rate)        
        
    def add_spontaneous(self, source, target, rate):
        self.transitions.add_edge(source, target, rate=rate)
        
    def _new_cases(self, population, time, pos):
        """Internal function used by integration routine"""
        diff = np.zeros(len(pos))
        N = np.sum(population)        
        
        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]
            
            rate = trans['rate']*population[pos[source]]
            
            if 'agent' in trans:
                agent = trans['agent']
                rate *= population[pos[agent]]/N
                
            diff[pos[source]] -= rate
            diff[pos[target]] += rate
            
        return diff
    
    def plot(self, title=None, normed=True, **kwargs):
        """Convenience function for plotting"""
        try:
            if normed:
                N = self.values_.iloc[0].sum()
                ax = (self.values_/N).plot(**kwargs)
            else:
                ax = self.values_.plot(**kwargs)
                
            ax.set_xlabel('Time')
            ax.set_ylabel('Population')
            
            if title is not None:
                ax.set_title(title)
            
            return ax
        except:
            raise NotInitialized('You must call integrate() first')
    
    def __getattr__(self, name):
        """Dynamic method to return the individual compartment values"""
        if 'values_' in self.__dict__:
            return self.values_[name]
        else:
            raise AttributeError("'EpiModel' object has no attribute '%s'" % name)

    def simulate(self, timesteps, **kwargs):
        """Stochastically simulate the epidemic model"""
        pos = {comp: i for i, comp in enumerate(kwargs)}
        population=np.zeros(len(pos), dtype='int')

        for comp in pos:
            population[pos[comp]] = kwargs[comp]

        values = []
        values.append(population)

        comps = list(self.transitions.nodes)
        time = np.arange(1, timesteps, 1, dtype='int')

        for t in time:
            pop = values[-1]
            new_pop = values[-1].copy()
            N = np.sum(pop)


            for comp in comps:
                trans = list(self.transitions.edges(comp, data=True))             

                prob = np.zeros(len(comps), dtype='float')

                for _, node_j, data in trans:
                    source = pos[comp]
                    target = pos[node_j]

                    rate = data['rate']

                    if 'agent' in data:
                        agent = pos[data['agent']]
                        rate *= pop[agent]/N

                    prob[target] = rate

                prob[source] = 1-np.sum(prob)

                delta = random.multinomial(pop[source], prob)
                delta[source] = 0

                changes = np.sum(delta)

                if changes == 0:
                    continue

                new_pop[source] -= changes

                for i in range(len(delta)):
                    new_pop[i] += delta[i]

            values.append(new_pop)

        values = np.array(values)
        self.values_ = pd.DataFrame(values[1:], columns=comps, index=time)
    
    def integrate(self, timesteps, **kwargs):
        """Numerically integrate the epidemic model"""
        pos = {comp: i for i, comp in enumerate(kwargs)}
        population=np.zeros(len(pos))
        
        for comp in pos:
            population[pos[comp]] = kwargs[comp]
        
        time = np.arange(1, timesteps, 1)

        self.values_ = pd.DataFrame(scipy.integrate.odeint(self._new_cases, population, time, args=(pos,)), columns=pos.keys(), index=time)

    def __repr__(self):
        text = 'Epidemic Model with %u compartments and %u transitions:\n\n' % \
              (self.transitions.number_of_nodes(), 
               self.transitions.number_of_edges())
        
        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]
            
            rate = trans['rate']

            if 'agent' in trans:
                agent = trans['agent']
                text += "%s + %s = %s %f\n" % (source, agent, target, rate)
            else:
                text+="%s -> %s %f\n" % (source, target, rate)
                
        return text


if __name__ == '__main__':

    SIR = EpiModel()
    SIR.add_interaction('S', 'I', 'I', 0.2)
    SIR.add_spontaneous('I', 'R', 0.1)

    N = 100000
    fig, ax = plt.subplots(1)

    values = []
    Nruns = 100

    for i in range(Nruns):
        SIR.simulate(365, S=N-1, I=1, R=0)
        ax.plot(SIR.I/N, lw=.1, c='b')
        if SIR.I.max() > 10:
            values.append(SIR.I)

    ax.set_xlabel('Time')
    ax.set_ylabel('Population')

    values =  pd.DataFrame(values).T
    values.columns = np.arange(values.shape[1])
    ax.plot(values.median(axis=1)/N, lw=1, c='r')
    fig.savefig('SIR.png')
