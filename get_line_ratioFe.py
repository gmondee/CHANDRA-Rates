import pyatomdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

plt.ion()
# The results of this look very flat, as the main thing that changes with temperature is meant to be the ionization
# balance, and I have essentially frozen that. 

### Load in ion balance pickle file
Z = 26
element = pyatomdb.atomic.Ztoelsymb(Z)
picklename = f'{element}_bal.pickle'
with open(os.path.abspath(os.path.join('pickle',picklename)), 'rb') as file:
  ionbal = pickle.load(file)




Telist = ionbal['TlistK'] #numpy.logspace(4,9,21) # big list o' temperatures



n  = pyatomdb.spectrum.NEISession(elements=[8])

taulist = 1e14 #(irrelevant)
emisE=np.zeros(len(Telist))
emisF=np.zeros(len(Telist))
emisG=np.zeros(len(Telist))
charges = list(range(9))

for i, T in enumerate(Telist):
  # define initial population here
  init_pop = {}
  init_pop[Z] = np.zeros(Z+1)

  ### Get ion balance at this temperature
  for c in charges:
    init_pop[Z][c]=ionbal[c][i]

  # freeze_ion_pop makes it calculate the emissivity based on the population you provide, as opposed to anything else.
  z1 = 7 # ion
  up = 7 # upper level of resonance line
  lo = 1 # lower level of resonance line
  e = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 


  # now change the ion charge to 8 and get the H-like emissivity. Need to sum 2 lines.
  z1 = 8 # ion
  up = 3 # upper level of resonance line
  f = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 

  up = 4 # upper level of resonance line
  g = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 

  emisE[i]=e['epsilon']
  emisF[i]=f['epsilon']
  emisG[i]=g['epsilon']

# let's plot some graphs

fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)

ax1.loglog(Telist, emisE, label='He-like')
ax1.loglog(Telist, emisF+emisG, label='H-like')


ax2.loglog(Telist, emisE/(emisF+emisG))

ax1.legend(loc=0)
ax1.set_ylabel('Emissivity (ph cm^3 s^-1)')
ax2.set_ylabel('He/H-like')
 
ax2.set_xlabel('Temperature (K)')