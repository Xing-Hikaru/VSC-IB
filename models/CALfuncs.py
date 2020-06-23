import numpy as np

# %%
def jac_f(mdl_func, x0, e=1e-8):
    m = mdl_func(x0).size
    n = x0.size
    jm = np.mat(np.zeros([m, n]))
    for i in range(n):
        ex = np.zeros(n, float)
        ex[i] = e
        jm[:][i] = (mdl_func(x0 + ex) - mdl_func(x0)) / e
    return jm

