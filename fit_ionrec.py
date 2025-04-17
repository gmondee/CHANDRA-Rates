import lmfit
import numpy as np
import matplotlib.pyplot as plt

KBOLTZ = 8.617385e-8
ME_KEV = 510.99895

def younger(electronEnergy, eion, A,B,C,D,E):
  #collisional ionization formula
  u = electronEnergy/eion
  epsilon = electronEnergy/ME_KEV
  tau = eion/ME_KEV
  R = ((tau+2)/(epsilon+2))* ( (epsilon+1)/(tau+1))**2 * \
       ( ((tau+epsilon)*(epsilon+2) *(tau+1)**2)/ (epsilon * (epsilon+2) * (tau+1)**2+tau*(tau+2)))**1.5

  Aterm = A*(1-(1/u))
  Bterm = B*(1-(1/u))**2
  Cterm = C* R * np.log(u)
  Dterm = D* np.log(u)/np.sqrt(u)
  Eterm = E* np.log(u)/u

  ci_cross = 1/(u*eion*eion)*(Aterm + Bterm + Cterm + Dterm + Eterm)
  ci_cross[u<1] = 0.0
  return ci_cross * 10**(2*2) #in units of cm^-2

def mewe(electronEnergy, eion, A,B,C,D,E):
  #excitation-autoionization formula
  u = electronEnergy/eion

  Aterm = A
  Bterm = B/u
  Cterm = C/(u**2)
  Dterm = 2*D/(u**3)
  Eterm = E*np.log(u)

  ea_cross = 1/(u*(eion**2))*(Aterm + Bterm + Cterm + Dterm + Eterm)# in units of 10^-24 m^2
  ea_cross[u<1] = 0.0
  return ea_cross * 10**(2*2) #in units of cm^-2