[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DataForScience/Epidemiology101/master)

# Epidemic Modeling for Everyone

![Multipopulation model](https://raw.githubusercontent.com/DataForScience/Epidemiology101/master/SEIIRD_model_season.png)

Repository to accompany the blog series: __Epidemic Modeling__

These blog posts were recently featured in the [Data Exchange Podcast](https://thedataexchange.media/) by [Ben Lorica](https://twitter.com/bigdata) in two episodes: 

- [Computational Models and Simulations of Epidemic Infectious Diseases](https://thedataexchange.media/computational-models-and-simulations-of-epidemic-infectious-diseases/)
	
- [Assessing Models and Simulations of Epidemic Infectious Diseases](https://thedataexchange.media/assessing-models-and-simulations-of-epidemic-infectious-diseases/)

## Background Information

<details>
<summary>Context-setting posts on CoVID-19 as a global phenomenon, no modeling required.</summary>

An introduction to the CoVID-19 pandemic and why it became the first truly global event of its kind, setting the stage for the modeling posts that follow.
</details>

1. [CoVID-19: Everything you need to know](https://data4sci.substack.com/p/covid-19-everything-you-need-to-know)

2. [CoVID-19: The first truly global event](https://data4sci.substack.com/p/covid-19-the-first-truly-global-event)

## Visualization

<details>
<summary>Notebooks that visualize CoVID-19 case, patient, and mortality data without building predictive models.</summary>

Covers plotting the geographic and temporal spread of the pandemic, exploring individual patient-level data, and building simple death-toll forecasts from observed trends.
</details>

1. [Epidemiology001.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology001.ipynb) - [Visualizing the spread of CoVID-19](https://data4sci.substack.com/p/visualizing-the-spread-of-covid-19) 

2. [Epidemiology002.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology002.ipynb) - [Visualizing individual CoVID-19 patient data](https://data4sci.substack.com/p/covid-19-visualizing-individual-patient) 

3. [Epidemiology003.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology003.ipynb) - [CoVID-19: Forecasting the death toll](https://data4sci.substack.com/p/covid-19-forecasting-the-death-toll) 

## Compartmental Models

<details>
<summary>The core SIR/SEIR-family models: exponential fits, confidence intervals, seasonality, and competing strains.</summary>

Builds up classic compartmental epidemic models step by step, starting from why naive exponential fits mislead, then adding uncertainty quantification, seasonal forcing, and competition between multiple circulating strains.
</details>

1. [Epidemiology101.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology101.ipynb) - [Epidemic Modeling 101: Or why your CoVID19 exponential fits are wrong](https://data4sci.substack.com/p/epidemic-modeling-101-or-why-your)

2. [Epidemiology102.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology102.ipynb) - [Epidemic Modeling 102: All CoVID-19 models are wrong, but some are useful](https://data4sci.substack.com/p/all-covid-19-models-are-wrong-but) 

3. [Epidemiology103.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology103.ipynb) - [Epidemic Modeling 103: Adding confidence intervals and stochastic effects to your CoVID-19 Models](https://data4sci.substack.com/p/adding-confidence-intervals-and-stochastic) 

4. [Epidemiology104.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology104.ipynb) - [Epidemic Modeling 104: Impact of Seasonal effects on CoVID-19](https://data4sci.substack.com/p/impact-of-seasonal-effects-on-covid)

5. [Epidemiology105.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology105.ipynb) - [Epidemic Modeling 105: Competing CoVID-19 Strains](https://data4sci.substack.com/p/competing-covid-19-strains) 

# Network models

<details>
<summary>Moving beyond well-mixed populations to explicit contact networks, super-spreaders, and degree correlations.</summary>

Examines how the structure of who-contacts-whom shapes an outbreak, including the role of super-spreaders in contact tracing and how correlations between connected individuals' degrees affect spreading dynamics.
</details>

1. [Epidemiology 201.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology201.ipynb) - [Epidemiology 201: Network Structure, Super-spreaders and Contact Tracing](https://data4sci.substack.com/p/network-structure-super-spreaders)

2. [Epidemiology 202.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology202.ipynb) - [Epidemiology 202: Network Models, the effect of degree correlations](https://data4sci.substack.com/p/network-models)

# Advanced Models

<details>
<summary>Extensions that add real-world structure: vaccination, age, geography, demographics, and social contagion.</summary>

Covers more realistic model extensions, including the impact of vaccination campaigns, age-structured populations, metapopulation (multi-location) spreading, demographic processes like births and deaths, and an application of epidemic modeling to the spread of ideas as a "cognitive virus."
</details>

1. [Epidemiology 301.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology301.ipynb) - [Epidemiology 301: How to model the effects of vaccination](https://data4sci.substack.com/p/how-to-model-the-effects-of-vaccination)

2. [Epidemiology 302.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology302.ipynb) - [Epidemiology 302: The Impact of Age Structure on Epidemic Spreading](https://data4sci.substack.com/p/the-impact-of-age-structure-on-epidemic)

3. [Epidemiology 303.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology303.ipynb) - [Epidemiology 303: Metapopulation Models](https://data4sci.substack.com/p/meta-population-models)

4. [Epidemiology 304.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology304.ipynb) - [Epidemiology 304: Demographics](http://data4sci.substack.com/p/demographic-processes)

5. [Epidemiology 305.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology305.ipynb) - [Epidemiology 305: The Epidemiology of a Cognitive Virus](https://data4sci.substack.com/p/the-epidemiology-of-a-cognitive-virus)

# The `epidemik` package

<details>
<summary>The compartmental modeling engine that powers the notebooks in this repo, from simple SIR models to networks and metapopulations.</summary>

Starting with `Epidemiology304`, these notebooks build their models with [`epidemik`](https://github.com/DataForScience/epidemik), a companion Python package for simulating compartmental epidemic models. It lets you define arbitrary compartmental models from interaction (`S + I -> I + I`) and spontaneous (`I -> R`) transitions, integrate them deterministically or run seeded stochastic simulations with the same interface, and compute R<sub>0</sub> automatically via the next-generation matrix. It also supports vaccination campaigns, birth/death rates, seasonal forcing, age structure, and — through its `NetworkEpiModel` and `MetaEpiModel` classes — epidemics on contact networks and across coupled sub-populations.
</details>

- GitHub: [DataForScience/epidemik](https://github.com/DataForScience/epidemik)
- PyPI: [pypi.org/project/epidemik](https://pypi.org/project/epidemik/)
- Documentation: [epidemik.readthedocs.io](https://epidemik.readthedocs.io/)

```python
from epidemik import EpiModel

SIR = EpiModel(seed=1337)
SIR.add_interaction('S', 'I', 'I', beta=0.2)
SIR.add_spontaneous('I', 'R', mu=0.1)
```

# Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment, including `epidemik` and the rest of the dependencies listed in `pyproject.toml`. To install the dependencies and launch Jupyter:

```bash
uv sync
uv run jupyter lab
```

`pygraphviz` compiles against the system Graphviz library. On macOS with Homebrew:

```bash
brew install graphviz
CFLAGS="-I$(brew --prefix graphviz)/include" LDFLAGS="-L$(brew --prefix graphviz)/lib" uv sync
```

## Author

<table border="0">
 <tr>
        <td>
          <img src="data/bgoncalves.png" alt="Bruno Gonçalves" width="150" height="150" style="border-radius: 50%; object-fit: cover;">
        </td>
        <td>
          <h2>Bruno Gonçalves</h2>
          <h3>Data For Science, Inc.</h3>
          <p>
                        Web: <a href="http://www.data4sci.com/">www.data4sci.com</a><br>
                        Twitter/X: <a href="https://twitter.com/bgoncalves">@bgoncalves</a><br>
                        LinkedIn: <a href="https://www.linkedin.com/in/bmtgoncalves/">@bmtgoncalves</a><br>
                        Email: <a href="mailto:info@data4sci.com">info@data4sci.com</a><br>
                        Schedule a Call: <a href="https://data4sci.com/call">https://data4sci.com/call</a>
          </p>
        </td>
 </tr>
</table>