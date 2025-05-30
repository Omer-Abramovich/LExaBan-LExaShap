# Advancing Fact Attribution for Query Answering: Aggregate Queries and Novel Algorithms
This repository is the official repository for the paper "Advancing Fact Attribution for Query Answering: Aggregate Queries and Novel Algorithms"

## Extended version of the paper
Extended version of the paper is available under [Advancing_Fact_Attribution_for_Query_Answering.pdf](Advancing_Fact_Attribution_for_Query_Answering.pdf)
Queries for all of the datasets used in the paper are under [Queries](Queries)

## Prerequisites

Before you begin, please ensure you have the following installed:

- Python (>=3.11)
- For obtaining provenance expressions of output tuples from queries or of aggregate queries as well as prerequisites for that please refer to the [ProvSQL project
](https://github.com/PierreSenellart/provsql)

## Algorithms' implementation

Full implementation of the code is in this repository. 
* LExaBan for DNF Boolean formulas is available under [LExaBan](Algorithms/LExaBan/BanzhafCircuit.py)
* LExaShap for DNF Boolean formulas is available under [LExaShap](Algorithms/LExaShap/ShapleyCircuit.py)
* LExaBan for MAX semimodule expressions is available under [Max_LExaBan](Algorithms/Max_LExaBan/ArithmeticCircuit.py)
* The implementation of previous SOTA algorithms ExaBan,ExaShap, and AdaBan are available [here](https://github.com/Omer-Abramovich/AdaBan)

## Usage instructions
### SPJU
  * All algorithms for SPJU expect a **DNF lineage** represented as a list of sets.  
Each set corresponds to a **conjunction (AND clause)**, and the list represents a **disjunction (OR of clauses)**.

```python
# Example: (x1 AND x2) OR (x3)
dnf_lineage = [
    {"x1", "x2"},
    {"x3"}
]
```

### Aggregate Queries
* Algorithms for SUM/COUNT Aggregate queries expect to recieve a list of DNFs and coefficients
```python
# Example: 3 * ((x1 AND x2) OR (x3)) + 5 * ((x4))
BNP_lineage = [
    ([
        {"x1", "x2"},
        {"x3"}
    ], 3),
    ([
        {"x4"}
    ], 5)
]
```

* Algorithms for MAX/MIN Aggregate queries expect to recieve a list of (set, coefficient) pairs, where each set is a conjunction (a clause in the DNF). This is available due to their idempotence (see Section 4 in the paper).
```python
# Example: max of {3 if (x1 AND x2), 7 if (x3), 2 if (x4 AND x5)}
max_semimodule_lineage = [
    ({"x1", "x2"}, 3),
    ({"x3"}, 7),
    ({"x4", "x5"}, 2)
]
```

### Examples
  
* Examples on how to run experiments on our algorithms can be found [here](Notebooks/Experiments.ipynb)
* Small example including visualization of a decomposition tree can be found [here](Notebooks/DtreeVisualization.ipynb)
