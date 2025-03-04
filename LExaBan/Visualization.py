from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR', scalar_prob_grad = True):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        if scalar_prob_grad:
            dot.node(name=str(id(n)), label = "{" + n.label + "| prob %.4f | grad %.4f }" % (n.prob, n.grad), shape='record')
        else:
            dot.node(name=str(id(n)), label = "{" + n.label + "|" + str(n.grad) + " |" + str(n.prob) + "}", shape='record')

        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
      dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot