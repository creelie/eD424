import numpy as np
import sympy as sp

theta, t = sp.symbols('theta t', real=True, positive=True)
u, v = sp.symbols('u v', real=True, positive=True)
w, v2s = sp.symbols('w v2s', real=True, positive=True)
sqrt2 = sp.sqrt(2)

t_chk = 0.15
theta_W_chk = 2*np.arctan(np.cos(t_chk))
theta_chk = 0.5*(theta_W_chk + np.pi/2)

u0 = float(sp.tan(theta_chk/2))
v0 = float(sp.tan(t_chk/2))
w0 = (u0*(1+v0**2) - (1-v0**2))/(2*v0**2)
v2s0 = v0/(float(sqrt2)-1)

# check: does u_of_w_v(w0,v0) == u0 ?
u_of_w_v = ((1-v**2)+2*w*v**2)/(1+v**2)
u_check = float(u_of_w_v.subs({w:w0, v:v0}))
print("u0:", u0, "u_check:", u_check, "diff:", abs(u0-u_check))

# check: does substituting v=(sqrt2-1)*v2s at v2s0 reproduce v0?
v_check = float(((sqrt2-1)*v2s).subs({v2s: v2s0}))
print("v0:", v0, "v_check:", v_check, "diff:", abs(v0-v_check))

# Now check full composed u_final(w0, v2s0) against u0 directly
v_final = (sqrt2-1)*v2s
u_final = ((1-v_final**2)+2*w*v_final**2)/(1+v_final**2)
u_final_check = float(u_final.subs({w:w0, v2s:v2s0}))
print("u0:", u0, "u_final_check:", u_final_check, "diff:", abs(u0-u_final_check))

# check theta,t reconstruction from u0,v0
theta_check = float(2*sp.atan(u0))
t_check = float(2*sp.atan(v0))
print("theta_chk:", theta_chk, "theta_check:", theta_check, "diff:", abs(theta_chk-theta_check))
print("t_chk:", t_chk, "t_check:", t_check, "diff:", abs(t_chk-t_check))
