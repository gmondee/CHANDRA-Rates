import scipy
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

KBOLTZ = 8.617385e-8
ME_KEV = 510.99895

def fitCrossSections(Elist, CSdata, params):
  fit = curve_fit(younger, np.array(Elist), np.array(CSdata), p0=params['ci'][0][1:])
  return fit

def younger(energies, eion, A,B,C,D,E):
  #collisional ionization formula
  energies = np.array(energies)
  u = energies/eion
  epsilon = energies/ME_KEV
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

def mewe(energies, params):
  #excitation-autoionization formula
  energies = np.array(energies)
  eion = params[1]
  u = energies/eion
  A = params[2]
  B = params[3]
  C = params[4]
  D = params[5]
  E = params[6]

  Aterm = A
  Bterm = B/u
  Cterm = C/(u**2)
  Dterm = 2*D/(u**3)
  Eterm = E*np.log(u)

  ea_cross = 1/(u*(eion**2))*(Aterm + Bterm + Cterm + Dterm + Eterm)# in units of 10^-24 m^2
  ea_cross[u<1] = 0.0
  return ea_cross * 10**(2*2) #in units of cm^-2

def youngerPlusMewe(energies, params): #not right, abcde are different
  return younger(energies) + mewe(energies, params)