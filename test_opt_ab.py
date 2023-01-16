import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import scipy.sparse.linalg
import scipy.special

VERBOSE = True

a, b = -1, 1
n = 5
K = np.arange(2*n)

x = np.linspace(a, b, n)
w = (2/n)*np.ones(n)

I = (b**(K + 1) - a**(K + 1))/(K + 1)

def get_Jx(x, w):
    return np.block([[np.zeros(n)], [np.array([k*w*x**(k - 1) for k in K[1:]])]])

def get_Jw(x):
    return np.array([x**k for k in K])

def get_J(x, w):
    return np.block([get_Jx(x, w), get_Jw(x)])

print('optimizing Gaussian quadrature rule:')

maxtol = 1e-7
for iteration in range(100):
    M = np.array([w*x**k for k in K]).sum(1)
    R = I - M

    residual = np.sqrt((R**2).sum())
    if VERBOSE:
        print(f'  {iteration = }: {residual = }')
    if residual < 1e-15:
        break

    J = get_J(x, w)
    if VERBOSE:
        print(f'  * cond(J) = {np.linalg.cond(J)}')

    mul = lambda x: J.T@(J@x)
    Jt_J = scipy.sparse.linalg.LinearOperator(
        shape=(2*n, 2*n), matvec=mul, rmatvec=mul, matmat=mul, rmatmat=mul)
    _ = scipy.sparse.linalg.cg(Jt_J, J.T@R, tol=min(maxtol, residual))[0]

    # _ = scipy.sparse.linalg.cg(J.T@J, J.T@R, tol=min(maxtol, residual),
    #                            x0=np.zeros(2*n))[0]

    dx, dw = _[:len(x)], _[len(w):]

    x += dx
    w += dw

print(f'  done: # iters. = {iteration}, res. = {residual}')

x_gt, w_gt = scipy.special.roots_legendre(n)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x, w, s=10, c='k')
plt.xlim(-1, 1)
plt.subplot(1, 2, 2)
plt.semilogy(np.maximum(np.finfo(x.dtype).eps, abs(x - x_gt)), label=r'$|x - x_{gt}|$')
plt.semilogy(np.maximum(np.finfo(w.dtype).eps, abs(w - w_gt)), label=r'$|w - w_{gt}|$')
plt.legend()
plt.tight_layout()
plt.show()

print('results:')

p = np.polynomial.Polynomial(np.random.randn(2*n - 1))
P = p.integ(1)
p_I_gt = P(1) - P(-1)
p_I_quad = w@p(x)
rel_err = abs(p_I_gt - p_I_quad)/abs(p_I_gt)
print(f'* integral of p over [-1, 1] = {p_I_gt}')
print(f'* {n = }, {rel_err = }')
