import sympy as sp
import time

w, v2s = sp.symbols('w v2s', real=True, positive=True)
sqrt2 = sp.sqrt(2)

# Build a moderately complex rational expression in w,v2s with sqrt2
# coefficients, similar in spirit to what the pipeline produces, and
# compare cancel() with and without extension=True.
num = sp.expand((1 + sqrt2*w - v2s**2)**3 * (w**2 - sqrt2*v2s + 1)**2)
den = sp.expand((1 + sqrt2*w - v2s**2)**2 * (w + sqrt2*v2s)**3)
expr = num/den

t0 = time.time()
r1 = sp.cancel(expr)
print(f"plain cancel: {time.time()-t0:.2f}s, len {len(str(r1))}")

t0 = time.time()
r2 = sp.cancel(expr, extension=True)
print(f"cancel(extension=True): {time.time()-t0:.2f}s, len {len(str(r2))}")
print("match:", sp.simplify(r1-r2)==0)
