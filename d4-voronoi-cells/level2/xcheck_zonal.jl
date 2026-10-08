# Evaluate entries of Z_lambda with the evaluate_zonal_matrix of de Laat,
# Leijenhorst and de Muinck Keizer, reading the cache installed by
# setup_cache.sh, at fixed rational inner products.  Run from the folder of
# their package; compare the output with xcheck_zonal.py.
using LasserreSphericalCodes, Nemo
const L = LasserreSphericalCodes
u = [QQ(1,3), QQ(-2,7), QQ(1,5), QQ(-1,4), QQ(2,9), QQ(3,11)]
for (lam, w1, w2) in [([0,0],0,0), ([4,0],2,0), ([6,2],2,4), ([9,5],3,1),
                      ([10,4],4,2), ([11,3],7,3), ([14,0],14,0), ([12,2],10,0)]
    v = L.evaluate_zonal_matrix(4, lam, w1, w2, u; FF=QQ)
    println(lam[1], " ", lam[2], " ", w1, " ", w2, " ", numerator(v), " ", denominator(v))
end
