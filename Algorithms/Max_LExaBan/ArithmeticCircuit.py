from .PartialDistribution import PartialDistributionDict
from .HelperFunctions import UnionFind, set_first
from .MaxBanzhafEngine import Value
from collections import defaultdict


class ArithmeticCircuit():

  def __init__(self, dnf):
        tmp_vars = set(item for clause in dnf for item in clause[0])
        tmp_mons = [clause[1] for clause in dnf]
        var_to_val = {v: ArithmeticCircuit.create_value_from_var(v) for v in tmp_vars}
        mon_to_val = {i: ArithmeticCircuit.create_value_from_mon(m) for i, m in enumerate(tmp_mons)}
        self.vars = list(var_to_val.values())
        self.dnf = lift_recursively([{mon_to_val[i],*[var_to_val[v] for v in clause[0]]} for i, clause in enumerate(dnf)])
        
        self.root = self.build_dtree()

  def build_dtree(self):
    root = ArithmeticCircuitNode(self.dnf, "root")
    root.recursively_expand()
    model_count = root.value.prob
    # print("model count", model_count)
    root.value.backward()
    # banz_vals = dict()
    # for v in self.vars:
    #   v.prob = PartialDistributionDict({-1: 1.0, 0: 0.0}, locked=True)
    #   v.forward()
    #   banz_vals[v.label] = sum([(v - root.value.prob.get(k,0))*k for k,v in model_count.items() if k!= -1])
    #   v.prob = PartialDistributionDict({-1: 0.5, 0: 0.5}, locked=True)
    # root.value.prob = model_count
    # power = 1
    # self.banzhaf_values = {k: v* power for k,v in banz_vals.items()}
    self.banzhaf_values = {v.label: v.prob[0] * v.grad[0] for v in self.vars}
    
    return root


  @staticmethod
  def create_value_from_var(var):
    prob = PartialDistributionDict({-1: 0.5, 0: 0.5}, locked=True)
    prob.lock()
    return Value(prob, label = str(var))

  @staticmethod
  def create_value_from_mon(mon):
    prob = PartialDistributionDict({-1: 0, mon: 1}, locked=True)
    prob.lock()
    return Value(prob, label = str(mon))



class ArithmeticCircuitNode():
  def __init__(self, dnf, op = "leaf", children=set(), parent=None, complex_lift = False):
        self.dnf = dnf
        self.op = op
        self.children = children
        self.parent = parent
        self.value = None
        self.complex_lift = complex_lift

  def recursively_expand(self):
    expansion_res = expand(self.dnf, self.complex_lift)
    if not expansion_res[0]:
      self.value = Value.N_mul(self.dnf[0])
      self.op = '*'
    else:
      self.children = [ArithmeticCircuitNode(child, complex_lift=self.complex_lift or expansion_res[2] == '&') for child in expansion_res[1]]
      self.op = expansion_res[2]
      for child in self.children:
        child.recursively_expand()
      self.perform_op()


  def perform_op(self):
    if self.op == "+":
      self.value = Value.N_add([child.value for child in self.children])
    elif self.op == "*":
      self.value = Value.N_mul([child.value for child in self.children])
    elif isinstance(self.op, Value):
      self.value = self.children[0].value.exc_or(self.children[1].value, self.op)
    elif self.op == '&':
      self.value = self.children[0].value * self.children[1].value
    else:
      print("problematic op", self.op)
      assert False

def ind_or(dnf):
    uf = UnionFind()
    for clause in dnf:
        x = set_first(clause)
        for idx, fact in enumerate(clause):
            if idx > 0:
              uf.union(fact, x)

    outputs = dict()
    for idx, clause in enumerate(dnf):
        k = uf.find(set_first(clause))
        if k in outputs:
            outputs[k].append(clause)
        else:
            outputs[k] = [clause]

    if len(outputs) > 1:
      return True, list(outputs.values())
    else:
      return False, []

def ind_and(dnf):
  candidates = set(dnf[0])
  for i in range(1, len(dnf)):
    candidates = candidates.intersection(dnf[i])
    if len(candidates) == 0:
      break
  if len(candidates) == 0:
    return False, []
  else:
    dnf = [(clause - candidates) for clause in dnf]
    return True, [[candidates], dnf]

def exc_or(dnf):
  counts = defaultdict(int)

  for clause in dnf:
    for fact in clause:
        counts[fact] += 1
  max_fact = max(counts, key=counts.get)

  false_dnf = []
  tmp_dnf = []
  for clause in dnf:
    if max_fact not in clause:
      false_dnf.append(clause)
    else:
      tmp_dnf.append(clause - {max_fact})

  tmp_dnf = tmp_dnf + false_dnf
  tmp_dnf.sort(key=len)
  true_dnf = []

  # TODO - Bug. adress this logic change in cases where the clauses are subsets but the monoms aren't the same. maybe fixed?
  for clause in tmp_dnf:
    if not any(existing_set.issubset(clause) for existing_set in true_dnf):
      true_dnf.append(clause)

  return max_fact, [true_dnf, false_dnf]

def remove_monom(dnf):
  return False, []
  first_val = dnf[0][1]
  if first_val.label != 0:
    for clause in dnf:
      if clause[1] != first_val:
        return False, dnf
    return True, [[(clause[0], Value(PartialDistributionDict(int, {0 : 1}, locked=True), label= 0)) for clause in dnf], [({Value(PartialDistributionDict(int, {0 : 1}, locked=True), label= 0)}, first_val)]]
  else:
    return False, [dnf, (Value(1), first_val)] #TODO - chage Value(1) to special value


def expand(dnf, complex_lift = False):
  if len(dnf) == 1:
    return False, []

  res = ind_or(dnf)
  op = "+"
  children = res[1]
  if not res[0]:
    res = ind_and(dnf)
    op = "*"
    children = res[1]
  if not res[0]:
    res = remove_monom(dnf)
    op = "&"
    children = res[1]
  if not res[0]:
    res = exc_or(dnf)
    op = res[0]
    # children = res[1]
    children = [lift_recursively(child) for child in res[1]]


  return True, children, op


def lift_recursively(dnf):
  success = True
  while success and len(dnf) > 1:
    success, dnf = lift(dnf)

  return dnf

def lift(dnf):
  neighbors = defaultdict(set) #holding neighbors for each fact
  candidates = dict() #holding possible candidates for conjucntion for each fact
  for clause in dnf:
    for fact in clause:
      adjusted_clause = clause - {fact}
      neighbors[fact].add(frozenset(adjusted_clause))
      if fact in candidates:
        candidates[fact] &= adjusted_clause
      else:
        candidates[fact] = adjusted_clause

  lifted_dict_disjunction = defaultdict(list)
  for key, val in neighbors.items():
    lifted_dict_disjunction[frozenset(val)].append(key)

  lifted_dict_disjunction = list(filter(lambda x: len(x) > 1, lifted_dict_disjunction.values()))

  lifted_dict_conjuction = defaultdict(list)
  for key, val in candidates.items():
    for v in val:
      if v not in lifted_dict_conjuction and key in candidates[v]:
        lifted_dict_conjuction[key].append(v)

  lifted_dict_conjuction = [[k] + v for k,v in lifted_dict_conjuction.items()]

  #TODO - merge with similar code above
  replace = dict()
  delete = set()
  for group in lifted_dict_disjunction:
    new_val = Value.N_add(group)
    new_val.label = f"+{len(group)}" # for visualization, consider removing (TODO)
    replace[group[0]] = new_val
    delete |= set(group[i] for i in range(1, len(group)))

  new_dnf = []
  for clause in dnf:
    if all(fact not in delete for fact in clause):
      new_clause = {replace[fact] if fact in replace else fact for fact in clause}
      new_dnf.append(new_clause)

  replace = dict()
  delete = set()
  for group in lifted_dict_conjuction:
    new_val = Value.N_mul(group)
    new_val.label = f"*{len(group)}" # for visualization, consider removing (TODO)
    replace[group[0]] = new_val
    delete |= set(group[i] for i in range(1, len(group)))

  newer_dnf = []
  for clause in new_dnf:
    if any(fact in replace for fact in clause):
      new_clause = {replace[fact] if fact in replace else fact for fact in clause if fact not in delete}
      newer_dnf.append(new_clause)
    else:
      newer_dnf.append(clause)

  return bool(lifted_dict_disjunction or lifted_dict_conjuction), newer_dnf