# Advancing Fact Attribution for Query Answering: Aggregate Queries and Novel Algorithms
This repository is the official repository for the paper "Advancing Fact Attribution for Query Answering: Aggregate Queries and Novel Algorithms"

## Extended version of the paper
Extended version of the paper is available under [Advancing_Fact_Attribution_for_Query_Answering.pdf](Advancing_Fact_Attribution_for_Query_Answering.pdf)

## Prerequisites

Before you begin, please ensure you have the following installed:

- Python (>=3.11)
- For obtaining provenance expressions of output tuples from queries or of aggregate queries please refer to https://github.com/navefr/ShapleyForDbFacts

## Algorithms' implementation

Full implementation of the code is in this repository. 
* LExaBan and LExaShap for DNF formulas are available under BanzhafAlgorithms.py
* LExaBan for semimodule expressions is available under BanzhafAlgorithmsAggregates.py
* The implementation of previous SOTA algorithms ExaBan,ExaShap, and AdaBan are available https://github.com/Omer-Abramovich/AdaBan

## Experiments
* Example notebook on how to run experiments of our algorithms can be found [here](notebooks/Experiments.ipynb)
