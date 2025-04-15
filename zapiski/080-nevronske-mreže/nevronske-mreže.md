# Nevronske mreže

Tokratno poglavje začnemo s podatki. 

```python
n = 500
from sklearn.datasets import make_moons, make_blobs
X, ys = make_moons(n_samples=n, noise=0.1)
X0 = [x for x, y in zip(X, ys) if y == 0]
X1 = [x for x, y in zip(X, ys) if y == 1]
```

Podatke izrišimo v razsevnem diagramu.

![](lunice.svg)

Če podatke uporabimo za učenje logistične regresije, ta sicer poišče mejo med razredi, a z veliko izgubo, ki znaša okoli 0.3. Mejo med dvema razredoma za te podatke lahko sestavimo iz več linearnih odsekov, ki pa jih bi morali nekako zlepiti skupaj. Linearne odseke, vsaj v teoriji, lahko zgradimo z logistično regresijo, njih kombinacijo pa spet poiščemo s tem istim modelom. **Tako kombinacijo logističnih regresij imenujemo nevronska mreža**.

## Struktura modela

