from functools import reduce
import numpy as np
from Helper_functions import multiconvolve, deconvolve, compute_grad_for_conv_base_and_base_val
from math import factorial
from scipy.special import gammaln

# from graphviz import Digraph


class Value:
    """stores a single dnf variable and its gradient"""

    all_assignments_by_size_cache = dict()

    def __init__(self, prob, vars=set(), _children=(), _op="", label=""):
        if type(prob) == int or type(prob) == float:
            prob = np.array([0, prob], dtype=np.longdouble)

        self.prob = prob
        self.vars = vars
        self.grad = np.zeros(len(self.vars) + 1, dtype=np.longdouble)
        self._backward = lambda: None
        self._forward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.label = str(label)  # for printing purposes

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            1 - (1 - self.prob) * (1 - other.prob),
            self.vars.union(other.vars),
            (self, other),
            "V",
        )

        def _forward():
            out.prob = 1 - (1 - self.prob) * (1 - other.prob)
        
        out._forward = _forward

        def _backward():
            self.grad += (1 - other.prob) * out.grad
            other.grad += (1 - self.prob) * out.grad

        out._backward = _backward

        return out

    @staticmethod
    def N_add(values):
        values = [(v if isinstance(v, Value) else Value(v)) for v in values]
        all_assignments = multiconvolve([v.all_assignments_by_size() for v in values])
        unsatisfied_assignments = multiconvolve([v.all_assignments_by_size() - v.prob for v in values])
        out_prob = all_assignments - unsatisfied_assignments
        


        out = Value(
            out_prob,
            set.union(*[v.vars for v in values]),
            set(values),
            "V",
        )

        def _forward():
            all_assignments = multiconvolve([v.all_assignments_by_size() for v in values])
            unsatisfied_assignments = multiconvolve([v.all_assignments_by_size() - v.prob for v in values])
            out.prob = all_assignments - unsatisfied_assignments

        out._forward = _forward

        def _backward():
            for v in values:
                without_v = deconvolve(unsatisfied_assignments, v.all_assignments_by_size() - v.prob)
                without_v_out = np.pad(without_v, (0, len(out.prob) - len(without_v)))
                grad_multipliers = compute_grad_for_conv_base_and_base_val(conv_base= without_v, grad_size=len(v.prob), base_val=without_v_out)
                v.grad += np.dot(grad_multipliers, out.grad)

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.prob * other.prob, self.vars.union(other.vars), (self, other), "^"
        )

        def _forward():
            out.prob = self.prob * other.prob
        out._forward = _forward

        def _backward():
            self.grad += other.prob * out.grad
            other.grad += self.prob * out.grad

        out._backward = _backward

        return out

    @staticmethod
    def N_mul(values):
        values = [(v if isinstance(v, Value) else Value(v)) for v in values]

        out_prob = multiconvolve([v.prob for v in values])
        out = Value(
            out_prob,
            set.union(*[v.vars for v in values]),
            set(values),
            "*",
        )

        def _forward():
            out.prob = multiconvolve([v.prob for v in values])
        
        out._forward = _forward

        def _backward():
             for v in values:
                without_v = deconvolve(out.prob, v.prob)
                grad_multipliers = compute_grad_for_conv_base_and_base_val(conv_base= without_v, grad_size=len(v.prob), base_val=np.zeros(len(out.prob)))
                v.grad += np.dot(grad_multipliers, out.grad)

        out._backward = _backward

        return out

    def exc_or(self, other, ex_var):
        other = other if isinstance(other, Value) else Value(other)
        ex_var = ex_var if isinstance(ex_var, Value) else Value(ex_var)
        unsatisfied_ex_var = ex_var.all_assignments_by_size() - ex_var.prob
        true_branch_assignments = np.convolve(self.prob, ex_var.prob, mode='full')
        false_branch_assignments = np.convolve(other.prob, unsatisfied_ex_var, mode='full')
        out_prov = true_branch_assignments + false_branch_assignments
        
       
        out = Value(
            out_prov,
            self.vars.union(other.vars).union(ex_var.vars),
            (self, other, ex_var),
            "X" + ex_var.label,
        )

        def _forward():
            out.prob = np.pad(self.prob,(1,0)) * ex_var.prob[1] + other.prob * (1 - ex_var.prob[1])
            
        out._forward = _forward

        def _backward():
            self.grad += np.array([np.dot(np.pad(ex_var.prob, (i, len(out.prob) - i - len(ex_var.prob))), out.grad) for i in range(len(self.prob))], dtype=np.longdouble)
            other.grad += np.array([np.dot(np.pad(unsatisfied_ex_var, (i, len(out.prob) - i - len(ex_var.prob))), out.grad) for i in range(len(other.prob))], dtype=np.longdouble)
            ex_var.grad += np.array([np.dot(np.pad(self.prob, (i, len(out.prob) - i - len(self.prob))) - np.pad(other.prob, (i, len(out.prob) - i - len(other.prob))), out.grad) for i in range(len(ex_var.prob))], dtype=np.longdouble)

        out._backward = _backward

        return out

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

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.array([0,*[(factorial(i) * factorial(len(self.vars) - i - 1)) for i in range(len(self.vars))]], dtype=np.longdouble)
        # if len(self.vars) > 5000:
        #     self.grad = np.ones(len(self.vars) + 1, dtype=np.longdouble)
        # else:
        #     self.grad = np.array([0,*[(factorial(i) * factorial(len(self.vars) - i - 1)) for i in range(len(self.vars))]], dtype=np.longdouble)
        # n = len(self.vars)
        # log_factorials = np.array([0, *[(gammaln(i + 1) + gammaln(n - i)) for i in range(n)]], dtype=np.longdouble)
        # log_factorials -= np.max(log_factorials)
        # self.grad = np.exp(log_factorials, dtype=np.longdouble)

        for v in reversed(topo):
            v._backward()

    def forward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in topo:
          v._forward()

    def partial_tmp_forward(self, fact):
        topo = []
        visited = set()

        def build_topo(v):
            if (v not in visited) and (fact in v.vars):
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        

        prev_values = dict()
        for v in topo:
          prev_values[v] = v.prob
          v._forward()
        res = self.prob
        for v in topo:
            v.prob = prev_values[v]
        return res

    def zero_grad(self):
        self.grad = 0
        for v in self._prev:
            v.zero_grad()

    def update_value(self, new_val):
        self.prob = new_val.prob
        self.vars = new_val.vars
        self.grad = new_val.grad
        self._backward = new_val._backward
        self._prev = new_val._prev
        self._op = new_val._op
        self.label = new_val.label

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        # return f"{self.label}"
        return f"Value(label={self.label}, prob={self.prob}, grad={self.grad})"

    def all_assignments_by_size(self):
        return Value._all_assignments_by_size(len(self.vars))
    
    @staticmethod
    def _all_assignments_by_size(n):
        if n in Value.all_assignments_by_size_cache:
            return Value.all_assignments_by_size_cache[n]
        else:
            two_power_n = 2**n
            result = [0] * (n + 1)

            # Compute probabilities iteratively
            current = 1 / two_power_n  # Start with C(n, 0) / 2^n
            result[0] = current

            for i in range(1, n + 1):
                current *= (n - i + 1) / i  # Use the recursive formula
                result[i] = current
            return np.array(result, dtype=np.longdouble)
        
        
