# Program demonstrira sestavljanje, izračun vrednosti spremenljivk in odvodov ter
# risanje računskega grafa za izraz L = (a * b + c) * d.

from mgrad_three import Value

a = Value(3, 'a')
b = Value(42, 'b')
print(a + 10)
print(-13 + a)
print(a - b)
print((a + b) ** 3)