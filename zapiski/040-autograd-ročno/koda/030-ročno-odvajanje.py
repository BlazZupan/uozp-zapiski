# Program demonstrira ročno odvajanje funkcije L(a, b, c, d) = (a * b + c) * d.
# Odvode smo izračunali z metodo končnih diferenc.

def L(a, b, c, d):
    e = a * b
    f = e + c
    return f * d

# Vrednosti parametrov
a, b, c, d = 2, -3, 10, -2

# Izračun vrednosti e in f
e = a * b
f = e + c
print(f"e = {e}")  # Pričakujemo -6
print(f"f = {f}")  # Pričakujemo 4

h = 0.001
df = (L(a, b, c, d + h) - L(a, b, c, d)) / h
print(f"∂L/∂f = {df}")  # Pričakujemo 4
dd = (L(a, b, c, d + h) - L(a, b, c, d)) / h
print(f"∂L/∂d = {dd}")  # Pričakujemo 4
dc = (L(a, b, c + h, d) - L(a, b, c, d)) / h
print(f"∂L/∂c = {dc}")  # Pričakujemo -2
de = (L(a + h, b, c, d) - L(a, b, c, d)) / h
print(f"∂L/∂e = {de}")  # Pričakujemo -2
da = (L(a + h, b, c, d) - L(a, b, c, d)) / h
print(f"∂L/∂a = {da}")  # Pričakujemo 6
db = (L(a, b + h, c, d) - L(a, b, c, d)) / h
print(f"∂L/∂b = {db}")  # Pričakujemo -4