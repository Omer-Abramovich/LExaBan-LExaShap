# Advancing Fact Attribution for Query Answering: Aggregate Queries and Novel Algorithms
This repository is the official repository for the paper "Advancing Fact Attribution for Query Answering: Aggregate Queries and Novel Algorithms"

## Extended version of the paper
Extended version of the paper is available under [Advancing_Fact_Attribution_for_Query_Answering.pdf](Advancing_Fact_Attribution_for_Query_Answering.pdf)

## Prerequisites

Before you begin, please ensure you have the following installed:

- Python (>=3.11)
- For obtaining provenance expressions of output tuples from queries or of aggregate queries as well as prerequisites for that please refer to the [ProvSQL project
](https://github.com/PierreSenellart/provsql)

## Algorithms' implementation

Full implementation of the code is in this repository. 
* LExaBan for DNF Boolean formulas is available under [LExaBan](LExaBan/BanzhafCircuit.py)
* LExaShap for DNF Boolean formulas is available under [LExaShap](LExaShap/ShapleyCircuit.py)
* LExaBan for MAX semimodule expressions is available under [Max_LExaBan](Max_LExaBan/ArithmeticCircuit.py)
* The implementation of previous SOTA algorithms ExaBan,ExaShap, and AdaBan are available [here](https://github.com/Omer-Abramovich/AdaBan)
