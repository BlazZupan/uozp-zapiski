# Program demonstrira uporabo graphviz za risanje grafa funkcije.

import graphviz

dot = graphviz.Digraph()
dot.attr(rankdir='LR')  # riši od leve proti desni

# Create subgraphs to control vertical positioning
with dot.subgraph() as s:
    s.attr(rank='source')
    s.node('data', 'podatki\n X, y', shape='circle', fixedsize='true', width='1.0')

with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('model', 'struktura\nmodela', shape='circle', fixedsize='true', width='1.0')
    s.node('optimization', 'optimizacija', shape='box')
    s.node('parameters', 'parametri\n modela\n&Theta;*', shape='circle', fixedsize='true', width='1.0', style='filled', fillcolor='lightblue')

with dot.subgraph() as s:
    s.attr(rank='sink')
    s.node('criterion', 'kriterijska\nfunkcija\n L(&Theta;)', shape='circle', fixedsize='true', width='1.0')

dot.edge('data', 'optimization')
dot.edge('model', 'optimization')
dot.edge('criterion', 'optimization')
dot.edge('optimization', 'parameters')

dot.render('ucenje', format='svg', cleanup=True)