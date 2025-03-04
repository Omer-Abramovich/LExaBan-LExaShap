import numpy as np
from PartialDistribution import PartialDistributionDict

class Value:
    """ stores a single dnf variable and its gradient"""

    def __init__(self, prob, vars = set(), _children=(), _op='', label = ''):
        self.prob = prob
        self.vars = vars
        self.grad = 0
        self._forward = lambda: None
        self._backward = lambda: None
        self._prev = _children
        self._parents = set()
        for child in self._prev:
          child._connect_parent(self)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.label = str(label) #for printing purposes
        self.grad = {}

    def _connect_parent(self, val):
      self._parents.add(val)


    @staticmethod
    def N_add(values):
      values = [(v if isinstance(v, Value) else Value(v)) for v in values]
      out = Value(
            PartialDistributionDict.max_of_distributions([v.prob for v in values]),
             set.union(*[v.vars for v in values]),
            set(values), "V"
        )

      def _forward():
            out.prob = PartialDistributionDict.max_of_distributions([v.prob for v in values])
      out._forward = _forward

      def _backward():
            for idx, v in enumerate(values):
                grad_multipliers = out.prob.children_grad_multipliers[idx]
                for key ,multipliers in grad_multipliers.items():
                    if key > -1:
                        if key not in v.grad:
                            v.grad[key] = 0
                        for k, m in multipliers.items():
                            if k > -1:
                                v.grad[key] += out.grad.get(k,0) * m

      out._backward = _backward

      return out

    @staticmethod
    def N_mul(values):
      if len(values) == 1:
          out = list(values)[0]
          return out
      values = [(v if isinstance(v, Value) else Value(v)) for v in values]
      for v in values:
          v.prob.positive = True
         
      out = Value(
           PartialDistributionDict.max_of_distributions([v.prob for v in values]),
             set.union(*[v.vars for v in values]),
            set(values), "*"
        )
      
      for v in values:
          v.prob.positive = False

      def _forward():
            for v in values:
                v.prob.positive = True
            out.prob = PartialDistributionDict.max_of_distributions([v.prob for v in values])
            for v in values:
                v.prob.positive = False
      out._forward = _forward

      def _backward():
            for idx, v in enumerate(values):
                grad_multipliers = out.prob.children_grad_multipliers[idx]
                for key ,multipliers in grad_multipliers.items():
                    if key > -1:
                        if key not in v.grad:
                            v.grad[key] = 0
                        for k, m in multipliers.items():
                            if k > -1:
                                v.grad[key] += out.grad.get(k,0) * m

      out._backward = _backward

      return out

    def exc_or(self, other, ex_var):
        other = other if isinstance(other, Value) else Value(other)
        ex_var = ex_var if isinstance(ex_var, Value) else Value(ex_var)

        out = Value(PartialDistributionDict.weighted_distribution(self.prob, other.prob, ex_var.prob), self.vars.union(other.vars).union(ex_var.vars), (self, other, ex_var), 'X' + ex_var.label)


        def _forward():
            out.prob = PartialDistributionDict.weighted_distribution(self.prob, other.prob, ex_var.prob)
        out._forward = _forward

        def _backward():
            for idx, v in enumerate([self, other, ex_var]):
                grad_multipliers = out.prob.children_grad_multipliers[idx]
                for key ,multipliers in grad_multipliers.items():
                    if key not in v.grad:
                        v.grad[key] = 0
                    for k, m in multipliers.items():
                        v.grad[key] += out.grad.get(k,0) * m

        out._backward = _backward

        return out

    def forward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        pre_state = []
        for v in reversed(topo):
          pre_state.append(v.prob)
          v._forward()

        for idx, v in enumerate(reversed(topo)):
          if v._parents:
            v.prob = pre_state[idx]
        
    def backward(self):
            # topological order all of the children in the graph
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)

            self.grad = {i: i for i in self.prob if (i > -1 and self.prob[i] > 0)}

            # go one variable at a time and apply the chain rule to get its gradient
            for v in reversed(topo): 
                v._backward()

    def __repr__(self):
        return f"Value(label={self.label}, prob={self.prob}, grad={self.grad})"