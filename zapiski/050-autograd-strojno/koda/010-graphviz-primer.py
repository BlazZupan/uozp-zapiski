# Program demonstrira uporabo graphviz za risanje grafa funkcije.

import graphviz

dot = graphviz.Digraph()
dot.attr(rankdir='LR')  # riši od leve proti desni
dot.node('A')
dot.node('B')
dot.edge('A', 'B')
dot.node('C')
dot.edge('C', 'B')
dot.node('D')
dot.edge('B', 'D')
dot.render('enostavni-graf', format='svg', cleanup=True)