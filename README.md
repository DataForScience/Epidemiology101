# Epidemic Modeling for Everyone

![Multipopulation model](https://raw.githubusercontent.com/DataForScience/Epidemiology101/master/SEIIRD_model_season.png)

Repository to accompany the blog series: __Epidemic Modeling__

These blog posts were recently featured in the [Data Exchange Podcast](https://thedataexchange.media/) by [Ben Lorica](https://twitter.com/bigdata) in two episodes: 

- [Computational Models and Simulations of Epidemic Infectious Diseases](https://thedataexchange.media/computational-models-and-simulations-of-epidemic-infectious-diseases/)
	
- [Assessing Models and Simulations of Epidemic Infectious Diseases](https://thedataexchange.media/assessing-models-and-simulations-of-epidemic-infectious-diseases/)

## The `epidemik` package

The compartmental modeling engine that powers the notebooks in this repo, from simple SIR models to networks and metapopulations.

These notebooks build their models with [`epidemik`](https://github.com/DataForScience/epidemik), a companion Python package for simulating compartmental epidemic models. It lets you define arbitrary compartmental models from interaction (`S + I -> I + I`) and spontaneous (`I -> R`) transitions, integrate them deterministically or run seeded stochastic simulations with the same interface, and compute R<sub>0</sub> automatically via the next-generation matrix. It also supports vaccination campaigns, birth/death rates, seasonal forcing, age structure, and — through its `NetworkEpiModel` and `MetaEpiModel` classes — epidemics on contact networks and across coupled sub-populations.

Under the hood, a model is represented as a directed multigraph (built on `networkx`), integrated with `scipy`'s ODE solvers, and returned as tidy `pandas` DataFrames that plug directly into `matplotlib` for quick trajectory and model-structure plots. Named parameters can reference one another as expressions (e.g. `mu="beta/2"`), which keeps related rates in sync as you sweep scenarios. Model definitions can be saved to and loaded from YAML, so you can version-control a model independently of the notebook that runs it, or pull a ready-made model straight from the `epidemik` repository. Starting with `Epidemiology304`, every notebook in the Advanced Models section builds directly on top of `epidemik` rather than hand-rolled ODE code, so understanding the package is the fastest way to follow — and extend — those notebooks.


- GitHub: [DataForScience/epidemik](https://github.com/DataForScience/epidemik)
- PyPI: [pypi.org/project/epidemik](https://pypi.org/project/epidemik/)
- Documentation: [epidemik.readthedocs.io](https://epidemik.readthedocs.io/)

```python
from epidemik import EpiModel

SIR = EpiModel(seed=1337)
SIR.add_interaction('S', 'I', 'I', beta=0.2)
SIR.add_spontaneous('I', 'R', mu=0.1)
```


## Background Information

Context-setting posts on CoVID-19 as a global phenomenon, no modeling required.

An introduction to the CoVID-19 pandemic and why it became the first truly global event of its kind, setting the stage for the modeling posts that follow. The first post walks through the basics anyone needed to understand CoVID-19 as it was unfolding — how it spread, why it was different from prior outbreaks, and the terminology used throughout the rest of the series. The second post zooms out to explain what made this pandemic historically unusual: near-simultaneous, worldwide transmission enabled by modern travel networks, and why that global reach makes CoVID-19 a uniquely rich case study for the modeling techniques covered later in this repository.


1. [CoVID-19: Everything you need to know](https://data4sci.substack.com/p/covid-19-everything-you-need-to-know)

2. [CoVID-19: The first truly global event](https://data4sci.substack.com/p/covid-19-the-first-truly-global-event)

## Visualization

Notebooks that visualize CoVID-19 case, patient, and mortality data without building predictive models.

Covers plotting the geographic and temporal spread of the pandemic, exploring individual patient-level data, and building simple death-toll forecasts from observed trends. `Epidemiology001` reconstructs how the outbreak spread across countries and over time directly from public case-count data, giving you an intuitive feel for the data before any model is introduced. `Epidemiology002` drills down from aggregate counts to individual patient records, showing how demographics and outcomes vary case by case. `Epidemiology003` uses simple trend extrapolation — no compartmental model yet — to forecast near-term deaths, illustrating both the appeal and the pitfalls of naive forecasting that the later Compartmental Models section addresses head-on.


1. [Epidemiology001.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology001.ipynb) - [Visualizing the spread of CoVID-19](https://data4sci.substack.com/p/visualizing-the-spread-of-covid-19) 

2. [Epidemiology002.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology002.ipynb) - [Visualizing individual CoVID-19 patient data](https://data4sci.substack.com/p/covid-19-visualizing-individual-patient) 

3. [Epidemiology003.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology003.ipynb) - [CoVID-19: Forecasting the death toll](https://data4sci.substack.com/p/covid-19-forecasting-the-death-toll) 

## Compartmental Models

The core SIR/SEIR-family models: exponential fits, confidence intervals, seasonality, and competing strains.

Builds up classic compartmental epidemic models step by step, starting from why naive exponential fits mislead, then adding uncertainty quantification, seasonal forcing, and competition between multiple circulating strains. `Epidemiology101` explains why fitting a raw exponential to early case counts overestimates growth and leads to bad predictions, motivating the shift to compartmental (SIR-style) models. `Epidemiology102` introduces those compartmental models properly, along with an honest discussion of their assumptions and limitations. `Epidemiology103` adds confidence intervals and stochastic effects, so a model's output is a distribution of plausible trajectories rather than a single deterministic curve. `Epidemiology104` incorporates seasonal forcing, showing how transmission rates that vary over the year reshape the epidemic curve and complicate long-term projections. `Epidemiology105` extends the framework to multiple competing strains, modeling how variants interact and compete for the same susceptible population.


1. [Epidemiology101.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology101.ipynb) - [Epidemic Modeling 101: Or why your CoVID19 exponential fits are wrong](https://data4sci.substack.com/p/epidemic-modeling-101-or-why-your)

2. [Epidemiology102.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology102.ipynb) - [Epidemic Modeling 102: All CoVID-19 models are wrong, but some are useful](https://data4sci.substack.com/p/all-covid-19-models-are-wrong-but) 

3. [Epidemiology103.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology103.ipynb) - [Epidemic Modeling 103: Adding confidence intervals and stochastic effects to your CoVID-19 Models](https://data4sci.substack.com/p/adding-confidence-intervals-and-stochastic) 

4. [Epidemiology104.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology104.ipynb) - [Epidemic Modeling 104: Impact of Seasonal effects on CoVID-19](https://data4sci.substack.com/p/impact-of-seasonal-effects-on-covid)

5. [Epidemiology105.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology105.ipynb) - [Epidemic Modeling 105: Competing CoVID-19 Strains](https://data4sci.substack.com/p/competing-covid-19-strains) 

## Network models

Moving beyond well-mixed populations to explicit contact networks, super-spreaders, and degree correlations.

Examines how the structure of who-contacts-whom shapes an outbreak, including the role of super-spreaders in contact tracing and how correlations between connected individuals' degrees affect spreading dynamics. `Epidemiology201` replaces the homogeneous-mixing assumption of earlier notebooks with an explicit contact network, showing how a small number of highly connected super-spreaders can dominate transmission and how contact tracing exploits that structure to contain outbreaks more efficiently than blanket interventions. `Epidemiology202` goes further by varying degree correlations — whether highly connected individuals tend to link to other highly connected individuals or to poorly connected ones — and shows how that single structural property changes epidemic thresholds and final outbreak size, even when the average number of contacts stays fixed.


1. [Epidemiology 201.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology201.ipynb) - [Epidemiology 201: Network Structure, Super-spreaders and Contact Tracing](https://data4sci.substack.com/p/network-structure-super-spreaders)

2. [Epidemiology 202.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology202.ipynb) - [Epidemiology 202: Network Models, the effect of degree correlations](https://data4sci.substack.com/p/network-models)

## Advanced Models

Extensions that add real-world structure: vaccination, age, geography, demographics, and social contagion.

Covers more realistic model extensions, including the impact of vaccination campaigns, age-structured populations, metapopulation (multi-location) spreading, demographic processes like births and deaths, and an application of epidemic modeling to the spread of ideas as a "cognitive virus."

1. [Epidemiology 301.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology301.ipynb) - [Epidemiology 301: How to model the effects of vaccination](https://data4sci.substack.com/p/how-to-model-the-effects-of-vaccination)

2. [Epidemiology 302.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology302.ipynb) - [Epidemiology 302: The Impact of Age Structure on Epidemic Spreading](https://data4sci.substack.com/p/the-impact-of-age-structure-on-epidemic)

3. [Epidemiology 303.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology303.ipynb) - [Epidemiology 303: Metapopulation Models](https://data4sci.substack.com/p/meta-population-models)

4. [Epidemiology 304.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology304.ipynb) - [Epidemiology 304: Demographics](http://data4sci.substack.com/p/demographic-processes)

5. [Epidemiology 305.ipynb](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology305.ipynb) - [Epidemiology 305: The Epidemiology of a Cognitive Virus](https://data4sci.substack.com/p/the-epidemiology-of-a-cognitive-virus)


## Setup

### GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/DataForScience/Epidemiology101)

The repo includes a [`.devcontainer`](.devcontainer/devcontainer.json) configuration, so you can run every notebook online without installing anything locally. Click the badge above (or use the "Code" → "Codespaces" button on GitHub) to launch a ready-to-use environment: it installs the Graphviz/GEOS/PROJ system libraries `pygraphviz` and `cartopy` need, installs [`uv`](https://docs.astral.sh/uv/), and runs `uv sync` automatically. Once the Codespace finishes building, open any `.ipynb` file and select the `.venv` kernel (or run `uv run jupyter lab` in the terminal) to start working.

### Gitpod

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/DataForScience/Epidemiology101)

An alternative to Codespaces: the [`.gitpod.yml`](.gitpod.yml) config does the same setup — installing the system libraries `pygraphviz`/`cartopy` need, installing `uv`, and running `uv sync` — on top of Gitpod's own workspace image. Click the badge above to launch it, then run `uv run jupyter lab` in the terminal (or open a notebook directly with the VS Code Jupyter extension, which is pre-installed).

### Local install

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