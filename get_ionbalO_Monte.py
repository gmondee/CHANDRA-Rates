import pyatomdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

plt.ion()
# example to get recombination rates and ionizatino balance

Z = 8
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

for chargeState, values in ratesForAllChargeStates.items():
  initChargeState = int(chargeState[-2])
  for i, T in enumerate(Telist):
    ionratesExp[i][initChargeState] = values['rates'][i]
    ionratesUncLower[i][initChargeState] = values['uncsLower'][i]
    ionratesUncUpper[i][initChargeState] = values['uncsUpper'][i]

# this is to limit reloading files, trust me
datacache = {}

ionratesUrdam = np.zeros((len(Telist), Z), dtype=float) #[[Z0+, Z1+, Z2+,...], [Z0+, Z1+, Z2+...]]
recrates = np.zeros((len(Telist), Z), dtype=float)


for z1 in range(1,Z+1):
  itmp, rtmp = pyatomdb.atomdb.get_ionrec_rate(Telist, Te_unit='K', Z=Z,z1=z1, extrap=True, datacache=datacache)
  
  ionratesUrdam[:,z1-1]=itmp
  recrates[:,z1-1]=rtmp

"""
Monte Carlo: randomly sample at each temperature to make a new rate
  Use this rate to make a new ion balance
  Store each ion balance, and use the mean and stdev as the result and error bar
"""
monteCarloLength = 1000
ionbalExps = np.zeros((monteCarloLength, len(Telist), Z+1)) #each element in ionbalExps is a 27x300 array of each ion (27) at each temperature step (300)
uncScale = np.mean([np.abs(ionratesExp-ionratesUncLower), np.abs(ionratesUncUpper-ionratesUncUpper)], axis=0)

def monteIonbal(ionrates, uncertainties, Recrates):
  tmpionbal = np.zeros((len(Telist), Z+1), dtype=float)
  # Shuffle data points by drawing from their uncertainties
  tmpionrates = np.copy(ionrates)
  monteionrates = np.random.normal(loc=tmpionrates, scale=uncScale) #assume errors are symmetrical
  for iT in range(len(Telist)):
    tmpionbal[iT,:] = pyatomdb.apec.solve_ionbal( monteionrates[iT,:], Recrates[iT,:])
  if np.sum(tmpionbal>1): #try again if there is >1 ion population somewhere
    tmpionbal = monteIonbal(ionrates, uncertainties, Recrates)
  return tmpionbal

for i in range(monteCarloLength):
  ionbalExps[i]=monteIonbal(ionratesExp, uncScale, recrates)

ionbalExp = np.median(ionbalExps, axis=0)
ionbalUpper = np.percentile(ionbalExps, 84, axis=0)
ionbalLower = np.percentile(ionbalExps, 16, axis=0)


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

### save results as pickle file
ionbalResults = {}
ionbalResults['TlistK'] = Telist
for z1 in range(1, Z+2):
  #ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(z1-1)
  charge=int(z1-1)
  ionbalResults[charge] = {'ionbal':ionbalExp[:,charge],'upper':ionbalUpper[:,z1-1],'lower':ionbalLower[:,z1-1]}

bal_picklename = f'{element}_bal.pickle'
if os.path.exists(os.path.abspath(os.path.join('pickle',bal_picklename))):
  fexists = input("Pickle file already exists. Overwrite? [y]/[n]:\t")
  if fexists=='y':
    with open(os.path.abspath(os.path.join('pickle',bal_picklename)), 'wb') as file:
      pickle.dump(ionbalResults, file)
      print(f"Saved pickle file to {os.path.abspath(os.path.join('pickle',bal_picklename))}")
  else:
    print("Pickle file already exists; did not overwrite.")
else:
  with open(os.path.abspath(os.path.join('pickle',bal_picklename)), 'wb') as file:
    pickle.dump(ionbalResults, file)
    print(f"Saved pickle file to {os.path.abspath(os.path.join('pickle',bal_picklename))}")