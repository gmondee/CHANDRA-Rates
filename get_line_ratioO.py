import pyatomdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

plt.ion()
# The results of this look very flat, as the main thing that changes with temperature is meant to be the ionization
# balance, and I have essentially frozen that. 

### Load in ion balance pickle file
Z = 8
element = pyatomdb.atomic.Ztoelsymb(Z)
picklename = f'{element}_bal.pickle'
with open(os.path.abspath(os.path.join('pickle',picklename)), 'rb') as file:
  ionbal = pickle.load(file)




Telist = ionbal['TlistK'] #numpy.logspace(4,9,21) # big list o' temperatures



n  = pyatomdb.spectrum.NEISession(elements=[8])

taulist = 1e14 #(irrelevant)
emisE=np.zeros(len(Telist))
emisEup=np.zeros(len(Telist))
emisElo=np.zeros(len(Telist))
emisF=np.zeros(len(Telist))
emisFup=np.zeros(len(Telist))
emisFlo=np.zeros(len(Telist))
emisG=np.zeros(len(Telist))
emisGup=np.zeros(len(Telist))
emisGlo=np.zeros(len(Telist))
charges = list(range(9))

for i, T in enumerate(Telist):
  # define initial population here
  init_pop = {}
  init_pop_upper = {}
  init_pop_lower = {}
  init_pop[Z] = np.zeros(Z+1)
  init_pop_upper[Z] = np.zeros(Z+1)
  init_pop_lower[Z] = np.zeros(Z+1)

  ### Get ion balance at this temperature using the median ion balance and the uncertainties
  for c in charges:
    init_pop[Z][c]=ionbal[c]['ionbal'][i]
    init_pop_upper[Z][c]=ionbal[c]['upper'][i]
    init_pop_lower[Z][c]=ionbal[c]['lower'][i]

  # freeze_ion_pop makes it calculate the emissivity based on the population you provide, as opposed to anything else.
  z1 = 7 # ion
  up = 7 # upper level of resonance line
  lo = 1 # lower level of resonance line
  e = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 
  eup = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_upper, freeze_ion_pop=True, teunit='K') 
  elo = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_lower, freeze_ion_pop=True, teunit='K') 

  # now change the ion charge to 8 and get the H-like emissivity. Need to sum 2 lines.
  z1 = 8 # ion
  up = 3 # upper level of resonance line
  f = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 
  fup = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_upper, freeze_ion_pop=True, teunit='K') 
  flo = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_lower, freeze_ion_pop=True, teunit='K') 

  up = 4 # upper level of resonance line
  g = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 
  gup = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_upper, freeze_ion_pop=True, teunit='K') 
  glo = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_lower, freeze_ion_pop=True, teunit='K') 

  emisE[i]=e['epsilon']
  emisEup[i]=eup['epsilon']
  emisElo[i]=elo['epsilon']
  emisF[i]=f['epsilon']
  emisFup[i]=fup['epsilon']
  emisFlo[i]=flo['epsilon']
  emisG[i]=g['epsilon']
  emisGup[i]=gup['epsilon']
  emisGlo[i]=glo['epsilon']

# let's plot some graphs
cmap = plt.get_cmap("hsv", 3)
fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)

ax1.loglog(Telist, emisE, label='He-like', color=cmap(0))
ax1.fill_between(Telist, emisElo, emisEup, color=cmap(0), alpha=0.15)
ax1.loglog(Telist, emisF+emisG, label='H-like', color=cmap(1))
ax1.fill_between(Telist, emisFlo+emisGlo, emisFup+emisGup, color=cmap(1), alpha=0.15)

ax2.loglog(Telist, emisE/(emisF+emisG), cmap(0))
ax2.fill_between(Telist, emisElo/(emisFup+emisGup), emisEup/(emisFlo+emisGlo), color=cmap(0), alpha=0.15) #error bars are using low-H/high-He or vice versa

ax1.legend(loc=0)
ax1.set_xlim(2e6,4e8)
ax1.set_ylim(1e-22, 1e-14)
ax2.set_ylim(1e-4,1.1)
ax1.set_ylabel('Emissivity (ph cm^3 s^-1)')
ax2.set_ylabel('He/H-like')
 
ax2.set_xlabel('Temperature (K)')