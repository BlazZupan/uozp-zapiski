# Program demonstrira gradientni sestop za minimizacijo kvadratne funkcije f(a) = a^2 - 10a + 28,
# kjer smo odvod funkcije izračunali analitično.

def f(a):
    """Kvadratna funkcija, katere minimum iščemo."""
    return a**2 - 10*a + 28

def df(a):
    """Odvod funkcije f(a) = a^2 - 10a + 28."""
    return 2*a - 10

# Začetna vrednost parametra
a = 6
# Hitrost učenja
eta = 0.1

# Iterativni gradientni sestop
for _ in range(20):
    grad = df(a)
    a = a - eta * grad
    print(f"a = {a:.6f}, f(a) = {f(a):.6f}")
