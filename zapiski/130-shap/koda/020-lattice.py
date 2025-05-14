from itertools import combinations
from graphviz import Digraph

# Definicija atributov in njihovih imen
features = ['A', 'B', 'C']

# Vrednosti modela za vse kombinacije (v tisoč EUR)
model_values = {
    frozenset(): 100,
    frozenset(['A']): 80,
    frozenset(['B']): 50,
    frozenset(['C']): 130,
    frozenset(['A', 'B']): 40,
    frozenset(['A', 'C']): 90,
    frozenset(['B', 'C']): 80,
    frozenset(['A', 'B', 'C']): 90
}

# model_values = {
#     frozenset(): 100,
#     frozenset(['A']): 150,
#     frozenset(['B']): 140,
#     frozenset(['C']): 160,
#     frozenset(['A', 'B']): 180,
#     frozenset(['A', 'C']): 190,
#     frozenset(['B', 'C']): 200,
#     frozenset(['A', 'B', 'C']): 250
# }

# Funkcija za pretvorbo množice v ime
def set_name(s):
    return ''.join(sorted(s)) if s else '∅'

# Ustvari graf
dot = Digraph(comment='SHAP lattice')
dot.attr(rankdir='TB')
dot.attr('node', shape='circle', style='filled', fillcolor='lightgray')

# Dodaj vozlišča po nivojih
levels = {}
for s, value in model_values.items():
    name = set_name(s)
    label = f"{name}\n{value}"
    dot.node(name, label=label)
    levels.setdefault(len(s), []).append(name)

# Uporabi subgraphe za poravnavo nivojev
for k, nodes in levels.items():
    with dot.subgraph() as sub:
        sub.attr(rank='same')
        for node in nodes:
            sub.node(node)

# Robovi z razlikami (delta)
for size in range(3):
    for subset in combinations(features, size):
        subset_set = frozenset(subset)
        for f in features:
            if f not in subset_set:
                superset = subset_set.union([f])
                if superset in model_values:
                    src = set_name(subset_set)
                    dst = set_name(superset)
                    delta = model_values[superset] - model_values[subset_set]
                    dot.edge(src, dst, label=f"+{delta}" if delta > 0 else f"{delta}")

# Prikaz grafa
dot.render('shap_lattice', format='svg', view=True)
