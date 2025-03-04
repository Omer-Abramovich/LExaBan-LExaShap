class KeyDefaultDict(dict):
    def __missing__(self, key):
      return key

class UnionFind:
    def __init__(self):
        self.parent = KeyDefaultDict()

    def find(self, x):
        if self.parent[x] is None:
            return None
        path = []
        while x != self.parent[x]:
            path.append(x)
            x = self.parent[x]
        for node in path:
            self.parent[node] = x
        return x

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root is not None and y_root is not None and x_root != y_root:
            self.parent[x_root] = y_root


def set_first(s):
  for e in s:
    return e
