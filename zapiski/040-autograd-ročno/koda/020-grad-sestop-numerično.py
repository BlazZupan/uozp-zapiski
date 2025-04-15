# Program demonstrira gradientni sestop za minimizacijo kvadratne funkcije f(a) = a^2 - 10a + 28,
# kjer smo odvod funkcije izračunali numerično.

def f(a):
    """Kvadratna funkcija, katere minimum iščemo."""
    return a**2 - 10*a + 28

def df(a, h=0.0001):
    """Numerični odvod funkcije f z metodo končnih diferenc."""
    return (f(a + h) - f(a)) / h

# Začetna vrednost parametra
a = 6
# Hitrost učenja
eta = 0.1
# Parameter za numerično odvajanje
h = 0.0001

# Iterativni gradientni sestop
for _ in range(20):
    grad = df(a, h)
    a = a - eta * grad
    print(f"a = {a:.6f}, f(a) = {f(a):.6f}")
