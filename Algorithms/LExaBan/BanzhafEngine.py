from functools import reduce

# from graphviz import Digraph


class Value:
    """stores a single dnf variable and its gradient"""

    def __init__(self, prob, vars=set(), _children=(), _op="", label=""):
        self.prob = prob
        self.vars = vars
        self.grad = 0
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
        out = Value(
            1 - reduce(lambda x, y: x * (1 - y.prob), values, Value(1)).prob,
            set.union(*[v.vars for v in values]),
            set(values),
            "V",
        )

        def _forward():
            out.prob = 1 - reduce(lambda x, y: x * (1 - y.prob), values, Value(1)).prob

        out._forward = _forward

        def _backward():
            one_minus_prob_out_times_grad = (1 - out.prob) * out.grad
            for v in values:
                if v.prob == 1:
                    v.grad = out.grad
                else:
                    v.grad += one_minus_prob_out_times_grad / (1 - v.prob)

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
        out = Value(
            reduce(lambda x, y: x * y.prob, values, Value(1)).prob,
            set.union(*[v.vars for v in values]),
            set(values),
            "*",
        )

        def _forward():
            out.prob = reduce(lambda x, y: x * y.prob, values, Value(1)).prob
        
        out._forward = _forward

        def _backward():
            p_times_grad = out.prob * out.grad
            for v in values:
                if v.prob != 0:
                    v.grad += p_times_grad / v.prob
                else:
                    v.grad = 0

        out._backward = _backward

        return out

    def exc_or(self, other, ex_var):
        other = other if isinstance(other, Value) else Value(other)
        ex_var = ex_var if isinstance(ex_var, Value) else Value(ex_var)

        out = Value(
            self.prob * ex_var.prob + other.prob * (1 - ex_var.prob),
            self.vars.union(other.vars).union(ex_var.vars),
            (self, other, ex_var),
            "X" + ex_var.label,
        )

        def _forward():
            out.prob = self.prob * ex_var.prob + other.prob * (1 - ex_var.prob)
            
        out._forward = _forward

        def _backward():
            self.grad += out.grad * ex_var.prob
            other.grad += out.grad * (1 - ex_var.prob)
            ex_var.grad += out.grad * (self.prob - other.prob)

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
        self.grad = 2 ** len(self.vars) if len(self.vars) < 100 else 1
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
