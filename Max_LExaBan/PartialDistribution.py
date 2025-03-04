from collections import defaultdict
from bisect import bisect_right
from math import prod



class PartialDistributionDict(dict):
    def __init__(self, *args, locked=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.st = dict()


        if locked:
            self.lock()
        self.positive = False
        self.grad_multipliers = None

    def lock(self):
        if -1 not in self or self[-1] == 0:
            self[-1] = 1 - sum(list(self.values()))

        tot = 0
        for k in self.keys(): #Notice that in Python 3.7 and above, the keys are saved in an ordered way (based on insertion order)
            self.st[k] = tot
            tot += self[k]

        self.sorted_keys = sorted(self.keys())

    def smaller_than(self, key, second_call = False):
        if self.positive and not second_call:
            return self.positive_smaller_than(key)
        if key in self.st:
            return self.st[key]
        
        ind = bisect_right(self.sorted_keys, key)
        if ind == len(self.sorted_keys):
                self.st[key] = sum(v for v in self.values())
                return self.st[key]
        self.st[key] = self.st[self.sorted_keys[ind]]
        return self.st[self.sorted_keys[ind]]
        

    def positive_smaller_than(self, key):
        if key == -1:
            return 0
        return self.smaller_than(key, True) - self.get(-1, 0)


    @staticmethod
    def max_of_distributions(distributions):
        """
        Computes the distribution of the maximum value from a group of partial distributions.
        :param distributions: List of PartialDistribution instances.
        :return: A new PartialDistribution representing the max value distribution.
        """
        total_keys = sorted(set.union(*map(set, distributions)))
        max_dist = PartialDistributionDict()
        prod_without_key_d = [{} for _ in distributions]
        prod_with_key_d = [{} for _ in distributions]

        for key in total_keys:
            smaller_than_values = [d.smaller_than(key) for d in distributions]
            prod_with_key = prod([smaller_than_val + d.get(key, 0 ) for d, smaller_than_val in zip(distributions, smaller_than_values)])
            prod_without_key = prod(smaller_than_values)
            max_dist[key] = prod_with_key - prod_without_key
            for idx, d in enumerate(distributions):
                prod_without_key_d[idx][key] = prod_without_key / d.smaller_than(key) if prod_without_key != 0 else 0
                prod_with_key_d[idx][key] = prod_with_key / (d.smaller_than(key) + d.get(key,0)) if prod_with_key != 0 else 0
            
        grad_multipliers = [{} for _ in distributions]    
        for idx,d in enumerate(distributions):
            baseline_P_max =  {key: prod_with_key_d[idx][key] - prod_without_key_d[idx][key] for key in max_dist.keys()} if not d.positive else {key: 0 for key in max_dist.keys()}
            for k in d.keys():
                if k != -1 and d.get(k, 0) > 0:
                    grad_multipliers[idx][k] = {}
                    for key in max_dist.keys():
                        if k > key:
                            grad_multipliers[idx][k][key] = - baseline_P_max[key]
                        elif k == key:
                            grad_multipliers[idx][k][key] = prod_with_key_d[idx][key]  -  baseline_P_max[key]
                        elif k < key:
                            grad_multipliers[idx][k][key] = prod_with_key_d[idx][key] - prod_without_key_d[idx][key] - baseline_P_max[key]

        max_dist.children_grad_multipliers = grad_multipliers
        
        max_dist.lock()
        return max_dist
    
    @staticmethod
    def weighted_distribution(first, second, weights):
        """
        Computes the weighted distribution of the first and second by weights.
        """
        res = PartialDistributionDict()
        total_keys = sorted(set.union(*map(set, [first.keys(), second.keys()])))
        for key in total_keys:
            res[key] = first.get(key, 0) * weights.get(0, 0) + second.get(key, 0) * weights.get(-1, 0)
        res.lock()

        grad_multipliers = []
        grad_multipliers.append({p: {p: weights[0]} for p,v in first.items() if p != -1 and v > 0})
        grad_multipliers.append({p: {p: weights[-1]} for p,v in second.items() if p != -1 and v > 0})
        grad_multipliers.append({0: {k: first.get(k,0) - second.get(k,0) for k in total_keys}})

        res.children_grad_multipliers = grad_multipliers

        return res