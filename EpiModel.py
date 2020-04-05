### −∗− mode : python ; −∗−
# @file EpiModel.py
# @author Bruno Goncalves
######################################################

import networkx as nx
import numpy as np
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

    SIR.integrate(365, S=100000-1, I=1, R=0)
    SIR.plot()
    plt.gcf().savefig('SIR.png')
