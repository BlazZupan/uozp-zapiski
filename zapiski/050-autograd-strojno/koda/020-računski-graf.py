# Program demonstrira sestavljanje, izračun vrednosti spremenljivk in risanje računskega grafa
# za izraz L = (a * b + c) * d.

from mgrad_one import Value
from mplot import draw_dot

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = Value(-2.0, label='d')

e = a * b
e.label = 'e'
f = e + c
f.label = 'f'
L = f * d
L.label = 'L'

print(L)
graf = draw_dot(L)
graf.render('računski-graf', format='pdf', cleanup=True)
