### −∗− mode : python ; −∗−
# @file EpiModel.py
# @author Bruno Goncalves
######################################################

import networkx as nx
import numpy as np
from numpy import linalg
from numpy import random
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
import string

from tqdm import tqdm
tqdm.pandas()

class EpiModel(object):
    """Simple Epidemic Model Implementation
    
        Provides a way to implement and numerically integrate 
    """
    def __init__(self, compartments=None):
        """
        Initialize the EpiModel object
        
        Parameters:
        - compartments: list of strings, optional
            List of compartment names
        
        Returns:
        None
        """
        self.transitions = nx.MultiDiGraph()
        self.seasonality = None
        self.population = None
        self.orig_comps = None
        
        if compartments is not None:
            self.transitions.add_nodes_from([comp for comp in compartments])
    
    def add_interaction(self, source, target, agent, rate):  
        """
        Add an interaction between two compartments
        
        Parameters:
        - source: string
            Name of the source compartment
        - target: string
            Name of the target compartment
        - agent: string
            Name of the agent
        - rate: float
            Rate of the interaction
        
        Returns:
        None
        """      
        self.transitions.add_edge(source, target, agent=agent, rate=rate)        
        
    def add_spontaneous(self, source, target, rate):
        """
        Add a spontaneous transition between two compartments
        
        Parameters:
        - source: string
            Name of the source compartment
        - target: string
            Name of the target compartment
        - rate: float
            Rate of the transition
        
        Returns:
        None
        """
        self.transitions.add_edge(source, target, rate=rate)

    def add_vaccination(self, source, target, rate, start):
        """
        Add a vaccination transition between two compartments
        
        Parameters:
        - source: string
            Name of the source compartment
        - target: string
            Name of the target compartment
        - rate: float
            Rate of the vaccination
        - start: int
            Start time of the vaccination
        
        Returns:
        None
        """
        self.transitions.add_edge(source, target, rate=rate, start=start)

    def add_age_structure(self, matrix, population):
        self.contact = np.asarray(matrix)
        self.population = np.asarray(population).flatten()

        assert self.contact.shape[0] == self.contact.shape[1], "The contact matrix must be square." 

        age_groups = list(string.ascii_lowercase[:len(matrix)])
        n_ages = len(age_groups)

        model = EpiModel()
        self.orig_comps = list(self.transitions.nodes())

        for node_i, node_j, data in self.transitions.edges(data=True):
            # Interacting transition
            if "agent" in data:
                for i, age_i in enumerate(age_groups):
                    node_age_i = node_i + '_' + age_i
                    node_age_j = node_j + '_' + age_i

                    for j, age_j in enumerate(age_groups):
                        agent_age = data["agent"] + '_' + age_j

                        model.add_interaction(node_age_i, node_age_j, agent_age, data["rate"]*self.contact[i][j])

            # Spontaneous transition
            else:
                for age_i in age_groups:
                    node_age_i = node_i + '_' + age_i
                    node_age_j = node_j + '_' + age_i

                    if "start" not in data:
                        model.add_spontaneous(node_age_i, node_age_j, data["rate"])
                    else:
                        # vaccination
                        model.add_vaccination(node_age_i, node_age_j, data["rate"], data["start"])

        self.transitions = model.transitions
        
    def _new_cases(self, population, time, pos):
        """
        Internal function used by integration routine
        
        Parameters:
        - population: numpy array
            Current population of each compartment
        - time: float
            Current time
        - pos: dict
            Dictionary mapping compartment names to indices
        
        Returns:
        numpy array
            Array of new cases for each compartment
        """
        diff = np.zeros(len(pos))
        N = np.sum(population)

        if self.population is not None:
            N = {}

            for comp_i in self.transitions.nodes():
                age_group = comp_i.split('_')[-1]

                for comp_j in pos:
                    if comp_j.endswith(age_group):
                        N[comp_i] = N.get(comp_i, 0) + population[pos[comp_j]]
        
        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]
            
            rate = trans['rate']*population[pos[source]]
            
            if 'start' in trans and trans['start'] >= time:
                continue

            if 'agent' in trans:
                agent = trans['agent']

                if self.population is None:
                    rate *= population[pos[agent]]/N
                else:
                    rate *= population[pos[agent]]/N[source]

                if self.seasonality is not None:
                    curr_t = int(time)%365
                    season = float(self.seasonality[curr_t])
                    rate *= season
                
            diff[pos[source]] -= rate
            diff[pos[target]] += rate
            
        return diff
    
    def plot(self, title=None, normed=True, **kwargs):
        """
        Convenience function for plotting
        
        Parameters:
        - title: string, optional
            Title of the plot
        - normed: bool, optional
            Whether to normalize the values or not
        - kwargs: keyword arguments
            Additional arguments to pass to the plot function
        
        Returns:
        matplotlib.axes._subplots.AxesSubplot
            The plot object
        """
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
        """
        Dynamic method to return the individual compartment values
        
        Parameters:
        - name: string
            Name of the compartment
        
        Returns:
        pandas.Series
            The values of the specified compartment
        """        
        if 'values_' in self.__dict__:
            return self.values_[name]
        else:
            raise AttributeError("'EpiModel' object has no attribute '%s'" % name)

    def simulate(self, timesteps, t_min=1, seasonality=None, **kwargs):
        """
        Stochastically simulate the epidemic model
        
        Parameters:
        - timesteps: int
            Number of time steps to simulate
        - t_min: int, optional
            Starting time
        - seasonality: numpy array, optional
            Array of seasonal factors
        - kwargs: keyword arguments
            Initial population of each compartment
        
        Returns:
        None
        """
        pos = {comp: i for i, comp in enumerate(self.transitions.nodes())}
        population=np.zeros(len(pos), dtype='int')

        for comp in kwargs:
            population[pos[comp]] = kwargs[comp]

        values = []
        values.append(population)

        comps = list(self.transitions.nodes)
        time = np.arange(t_min, t_min+timesteps, 1, dtype='int')

        self.seasonality = seasonality

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

                    if 'start' in data and data['start'] >= t:
                        continue

                    if 'agent' in data:
                        agent = pos[data['agent']]
                        rate *= pop[agent]/N

                        if self.seasonality is not None:
                            curr_t = int(t)%365
                            season = float(self.seasonality[curr_t])
                            rate *= season

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
    
    def integrate(self, timesteps, t_min=1, seasonality=None, **kwargs):
        """
        Numerically integrate the epidemic model
        
        Parameters:
        - timesteps: int
            Number of time steps to integrate
        - t_min: int, optional
            Starting time
        - seasonality: numpy array, optional
            Array of seasonality values
        - kwargs: keyword arguments
            Initial population of each compartment
        
        Returns:
        None
        """        
        pos = {comp: i for i, comp in enumerate(self.transitions.nodes())}
        population=np.zeros(len(pos))
        total_pop = self.population.sum()

        for comp in kwargs:
            if self.population is None:
                if comp not in pos:
                    continue

                population[pos[comp]] = kwargs[comp]
            else:
                p = np.copy(self.population)/total_pop
                n = np.random.multinomial(kwargs[comp], p, 1)[0]

                for i, age in enumerate(string.ascii_lowercase[:len(p)]):
                    comp_age = comp + '_' + age
                    if comp_age not in pos:
                        continue

                    population[pos[comp_age]] = n[i]
        
        time = np.arange(t_min, t_min+timesteps, 1)

        self.seasonality = seasonality
        values = pd.DataFrame(scipy.integrate.odeint(self._new_cases, population, time, args=(pos,)), columns=pos.keys(), index=time)

        if self.population is None:
            self.values_ = values
        else:
            self.values_ages_ = values

            totals = values.T.copy()
            totals['key'] = totals.index.map(lambda x: '_'.join(x.split('_')[:-1]))
            totals = totals.groupby('key').sum().T
            totals.columns.name = None
            self.values_ = totals

    def __repr__(self):
        """
        Return a string representation of the EpiModel object
        
        Returns:
        string
            String representation of the EpiModel object
        """
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
            elif 'start' in trans:
                start = trans['start']
                text+="%s -> %s %f starting at %s days\n" % (source, target, rate, start)
            else:
                text+="%s -> %s %f\n" % (source, target, rate)
        
        R0 = self.R0()

        if R0 is not None:
            text += "\nR0=%1.2f" % R0

        return text

    def _get_active(self):
        active = set()

        for node_i, node_j, data in self.transitions.edges(data=True):
            if "agent" in data:
                active.add(data['agent'])
            else:
                active.add(node_i)

        return active

    def _get_susceptible(self):
        susceptible = set([node for node, deg in self.transitions.in_degree() if deg==0])

        if len(susceptible) == 0:
            for node_i, node_j, data in self.transitions.edges(data=True):
                if "agent" in data:
                    susceptible.add(node_i)

        return susceptible

    def _get_infections(self):
        inf = {}

        for node_i, node_j, data in self.transitions.edges(data=True):
            if "agent" in data:
                agent = data['agent']

                if agent not in inf:
                    inf[agent] = {}

                if node_i not in inf[agent]:
                    inf[agent][node_i] = {}

                inf[agent][node_i]['target'] = node_j
                inf[agent][node_i]['rate'] = data['rate']

        return inf


    def R0(self):
        infected = set()

        susceptible = self._get_susceptible()

        for node_i, node_j, data in self.transitions.edges(data=True):
            if "agent" in data:
                infected.add(data['agent'])
                infected.add(node_j)


        infected = sorted(infected)
        N_infected = len(infected)

        F = np.zeros((N_infected, N_infected), dtype='float')
        V = np.zeros((N_infected, N_infected), dtype='float')

        pos = dict(zip(infected, np.arange(N_infected)))

        try:
            for node_i, node_j, data in self.transitions.edges(data=True):
                rate = data['rate']

                if "agent" in data:
                    target = pos[node_j]
                    agent = pos[data['agent']]

                    if node_i in susceptible:
                        F[target, agent] = rate
                elif "start" in data:
                    continue
                else:
                    source = pos[node_i]

                    V[source, source] += rate

                    if node_j in pos:
                        target = pos[node_j]
                        V[target, source] -= rate
        
            eig, v = linalg.eig(np.dot(F, linalg.inv(V)))

            return eig.max()
        except:
            return None

    def __getitem__(self, bla):
        return self.values_[bla]

if __name__ == '__main__':

    Nk_uk = pd.read_csv("United Kingdom-2020.csv", index_col=0)
    Nk_ke = pd.read_csv("Kenya-2020.csv", index_col=0)

    contacts_uk = pd.read_excel("MUestimates_all_locations_2.xlsx", sheet_name="United Kingdom of Great Britain", header=None)
    contacts_ke = pd.read_excel("MUestimates_all_locations_1.xlsx", sheet_name="Kenya")

    beta = 0.05
    mu = 0.1

    SIR_uk = EpiModel()
    SIR_uk.add_interaction('S', 'I', 'I', beta)
    SIR_uk.add_spontaneous('I', 'R', mu)


    SIR_ke = EpiModel()
    SIR_ke.add_interaction('S', 'I', 'I', beta)
    SIR_ke.add_spontaneous('I', 'R', mu)

    N_uk = int(Nk_uk.sum())
    N_ke = int(Nk_ke.sum())


    SIR_uk.add_age_structure(contacts_uk, Nk_uk)
    SIR_ke.add_age_structure(contacts_ke, Nk_ke)

    SIR_uk.integrate(100, S=N_uk*.99, I=N_uk*.01, R=0)
    SIR_ke.integrate(100, S=N_ke*.99, I=N_ke*.01, R=0)

    fig, ax = plt.subplots(1)

    (SIR_uk['I']*100/N_uk).plot(ax=ax)
    (SIR_ke['I']*100/N_ke).plot(ax=ax)
    ax.legend(['UK', 'Kenya'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Population (%)')

    fig.savefig('SIR_age.png', dpi=300, facecolor='white')