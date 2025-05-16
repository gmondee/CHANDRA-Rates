import pyatomdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

plt.ion()
# example to get recombination rates and ionizatino balance

Z = 26
element = pyatomdb.atomic.Ztoelsymb(Z)
picklename = f'{element}.pickle'
with open(os.path.abspath(os.path.join('pickle',picklename)), 'rb') as file:
  ratesForAllChargeStates = pickle.load(file)

firstKey = list(ratesForAllChargeStates.keys())[0]
Telist = ratesForAllChargeStates[firstKey]['Tlist']
#Telist = np.logspace(4,9,300) #lots of temperatures

ionratesExp = np.zeros((len(Telist), Z), dtype=float)
ionratesUncLower = np.zeros((len(Telist), Z), dtype=float)
ionratesUncUpper = np.zeros((len(Telist), Z), dtype=float)
missingDataErrFrac = 0.2 #20% error for missing data

for chargeState, values in ratesForAllChargeStates.items():
  initChargeState = int(re.findall(r'\d+', chargeState)[0])
  for i, T in enumerate(Telist):
    ionratesExp[i][initChargeState] = values['rates'][i]
    ionratesUncLower[i][initChargeState] = values['uncsLower'][i]
    ionratesUncUpper[i][initChargeState] = values['uncsUpper'][i]

# this is to limit reloading files, trust me
datacache = {}

ionratesUrdam = np.zeros((len(Telist), Z), dtype=float) #[[Z0+, Z1+, Z2+,...], [Z0+, Z1+, Z2+...]] where each entry is at a different temperature
recrates = np.zeros((len(Telist), Z), dtype=float)


for z1 in range(1,Z+1):
  itmp, rtmp = pyatomdb.atomdb.get_ionrec_rate(Telist, Te_unit='K', Z=Z,z1=z1, extrap=True, datacache=datacache)
  
  ionratesUrdam[:,z1-1]=itmp
  recrates[:,z1-1]=rtmp

  ### fill in Urdam values where there isn't any data.
  if np.sum(ionratesExp[:,z1-1])==0:
    ionratesExp[:,z1-1]=itmp
    ionratesUncLower[:,z1-1] = (1-missingDataErrFrac)*itmp
    ionratesUncUpper[:,z1-1] = (1+missingDataErrFrac)*itmp



# at this point you would presumably swap in your own ionization rates for ionratesUrdamUrdam
ionbalExp = np.zeros((len(Telist), Z+1), dtype=float)
for iT in range(len(Telist)):
  ionbalExp[iT,:] = pyatomdb.apec.solve_ionbal( ionratesExp[iT,:], recrates[iT,:])

# try at the uncertainty extremes
ionbalLower = np.zeros((len(Telist), Z+1), dtype=float)
for iT in range(len(Telist)):
  ionbalLower[iT,:] = pyatomdb.apec.solve_ionbal( ionratesExp[iT,:]-ionratesUncLower[iT,:], recrates[iT,:])


ionbalUpper = np.zeros((len(Telist), Z+1), dtype=float)
for iT in range(len(Telist)):
  ionbalUpper[iT,:] = pyatomdb.apec.solve_ionbal( ionratesExp[iT,:]+ionratesUncUpper[iT,:], recrates[iT,:])


ionbalUrdam = np.zeros((len(Telist), Z+1), dtype=float)
for iT in range(len(Telist)):
  ionbalUrdam[iT,:] = pyatomdb.apec.solve_ionbal( ionratesUrdam[iT,:], recrates[iT,:])


# Plot some results
cmap = plt.get_cmap("hsv", Z+1)
fig = plt.figure()
ax = fig.add_subplot(111)

for z1 in range(1, Z+2):
  ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(z1-1)
  ax.loglog(Telist, ionbalUrdam[:,z1-1], linestyle='--', label=f'{ionsymb} Urd', color=cmap(z1-1))
  ax.loglog(Telist, ionbalExp[:,z1-1], linestyle='-', label=f'{ionsymb} Exp', color=cmap(z1-1))
  
ax.set_ylim(1e-3,1.3)
ax.legend(loc=0, ncol=2)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Ion Fraction')

# plot with errors
fig = plt.figure()
ax = fig.add_subplot(111)
for z1 in range(1, Z+2):
  ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(z1-1)
  ax.loglog(Telist, ionbalUrdam[:,z1-1], linestyle='--', linewidth=1, label=f'{ionsymb} Urd', color=cmap(z1-1))
  ax.loglog(Telist, ionbalExp[:,z1-1], linestyle='-', linewidth=1, label=f'{ionsymb} Exp', color=cmap(z1-1))
  ax.loglog(Telist, ionbalLower[:,z1-1], linestyle='-', linewidth=.5, color=cmap(z1-1))
  ax.loglog(Telist, ionbalUpper[:,z1-1], linestyle='-', linewidth=.5, color=cmap(z1-1))
  plt.fill_between(Telist, ionbalLower[:,z1-1], ionbalUpper[:,z1-1], color=cmap(z1-1), alpha=0.15)
  
ax.set_ylim(1e-3,1.3)
ax.legend(loc=0, ncol=2)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Ion Fraction')

for i in range(1, int((Z+1)/3)):  #plot in groups of 3
  baseZ = int((i-1)*3)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  for iz in [0,1,2]:
    z1 = baseZ+iz
    ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(z1-1)
    ax.loglog(Telist, ionbalUrdam[:,z1-1], linestyle='--', linewidth=1, label=f'{ionsymb} Urd', color=cmap(z1-1))
    ax.loglog(Telist, ionbalExp[:,z1-1], linestyle='-', linewidth=1, label=f'{ionsymb} Exp', color=cmap(z1-1))
    ax.loglog(Telist, ionbalLower[:,z1-1], linestyle='-', linewidth=.5, color=cmap(z1-1))
    ax.loglog(Telist, ionbalUpper[:,z1-1], linestyle='-', linewidth=.5, color=cmap(z1-1))
    plt.fill_between(Telist, ionbalLower[:,z1-1], ionbalUpper[:,z1-1], color=cmap(z1-1), alpha=0.15)
  ax.set_ylim(1e-3,1.3)
  ax.legend(loc=0, ncol=2)
  ax.set_xlabel('Temperature (K)')
  ax.set_ylabel('Ion Fraction')
#todo: uncertainties might not be working correctly for urdam data