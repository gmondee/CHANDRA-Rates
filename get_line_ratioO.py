import pyatomdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

plt.ion()
plt.rcParams.update({'font.size': 16})

### Load in ion balance pickle file
Z = 8
element = pyatomdb.atomic.Ztoelsymb(Z)
picklename = f'{element}_bal.pickle'
with open(os.path.abspath(os.path.join('pickle',picklename)), 'rb') as file:
  ionbal = pickle.load(file)




Telist = ionbal['TlistK'] #numpy.logspace(4,9,21) # big list o' temperatures



n  = pyatomdb.spectrum.NEISession(elements=[Z])

taulist = 1e14 #(irrelevant)
emisE=np.zeros(len(Telist))
emisEup=np.zeros(len(Telist))
emisElo=np.zeros(len(Telist))
emisEUrd=np.zeros(len(Telist))
emisF=np.zeros(len(Telist))
emisFup=np.zeros(len(Telist))
emisFlo=np.zeros(len(Telist))
emisFUrd=np.zeros(len(Telist))
emisG=np.zeros(len(Telist))
emisGup=np.zeros(len(Telist))
emisGlo=np.zeros(len(Telist))
emisGUrd=np.zeros(len(Telist))
charges = list(range(Z+1))

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
  z1 = Z-1 # ion
  up = 7 # upper level of resonance line
  lo = 1 # lower level of resonance line
  e = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 
  eup = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_upper, freeze_ion_pop=True, teunit='K') 
  elo = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_lower, freeze_ion_pop=True, teunit='K') 
  eUrd = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=False, teunit='K') 


  # now change the ion charge to 8 and get the H-like emissivity. Need to sum 2 lines.
  z1 = Z # ion
  up = 3 # upper level of resonance line
  f = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 
  fup = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_upper, freeze_ion_pop=True, teunit='K') 
  flo = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_lower, freeze_ion_pop=True, teunit='K') 
  fUrd = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=False, teunit='K') 


  up = 4 # upper level of resonance line
  g = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 
  gup = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_upper, freeze_ion_pop=True, teunit='K') 
  glo = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop_lower, freeze_ion_pop=True, teunit='K') 
  gUrd = n.return_line_emissivity(T, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=False, teunit='K') 


  emisE[i]=e['epsilon']
  emisEup[i]=eup['epsilon']
  emisElo[i]=elo['epsilon']
  emisEUrd[i]=eUrd['epsilon']
  emisF[i]=f['epsilon']
  emisFup[i]=fup['epsilon']
  emisFlo[i]=flo['epsilon']
  emisFUrd[i]=fUrd['epsilon']
  emisG[i]=g['epsilon']
  emisGup[i]=gup['epsilon']
  emisGlo[i]=glo['epsilon']
  emisGUrd[i]=gUrd['epsilon']


cmap = plt.get_cmap("hsv", 12)
fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)

ax1.loglog(Telist, emisEUrd, label='He-like AtomDB', linestyle='--', color=cmap(0))
ax1.loglog(Telist, emisE, label='He-like Experiment', color=cmap(1))
ax1.loglog(Telist, emisFUrd+emisGUrd, label='H-like AtomDB', linestyle='--', color=cmap(7))
ax1.loglog(Telist, emisF+emisG, label='H-like Experiment', color=cmap(6))
ax1.fill_between(Telist, emisFlo+emisGlo, emisFup+emisGup, color=cmap(6), alpha=0.15)
ax1.fill_between(Telist, emisElo, emisEup, color=cmap(1), alpha=0.15)

ax2.loglog(Telist, emisEUrd/(emisFUrd+emisGUrd), linestyle='--', color=cmap(10), label='AtomDB')
ax2.fill_between(Telist, emisElo/(emisFup+emisGup), emisEup/(emisFlo+emisGlo), color=cmap(10), alpha=0.15) #error bars are using low-H/high-He or vice versa
ax2.loglog(Telist, emisE/(emisF+emisG), color=cmap(9), label='Experiment')
ax2.grid(axis='y')

ax1.legend()
ax2.legend()
ax1.set_xlim(1e6,2e7)
ax1.set_ylim(1e-22, 1e-14)
ax2.set_ylim(1.5e-1,15)
ax1.set_ylabel('Emissivity (ph cm$^3$ s$^{-1}$)')
ax2.set_ylabel(f'{element}{Z-2}+/{element}{Z-1}+ Emission Ratio')
 
ax2.set_xlabel('Temperature (K)')
plt.title(f'{element}{Z-2}+ and {element}{Z-1}+ Emission')

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(f"{element}HHeRatio.pdf")
#for fig in range(1, plt.gcf().number + 1):
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.tight_layout()
pdf.savefig( fig )
pdf.close()
if False:
  # let's plot some graphs
  cmap = plt.get_cmap("hsv", 12)
  fig = plt.figure()
  fig.show()
  ax1 = fig.add_subplot(311)
  ax2 = fig.add_subplot(312, sharex=ax1)
  ax3 = fig.add_subplot(313, sharex=ax1)

  ax1.loglog(Telist, emisEUrd, label='He-like AtomDB', linestyle='--', color=cmap(1))
  ax1.loglog(Telist, emisE, label='He-like', color=cmap(0))
  ax1.fill_between(Telist, emisElo, emisEup, color=cmap(0), alpha=0.15)

  ax1.loglog(Telist, emisFUrd+emisGUrd, label='H-like AtomDB', linestyle='--', color=cmap(7))
  ax1.loglog(Telist, emisF+emisG, label='H-like', color=cmap(6))
  ax1.fill_between(Telist, emisFlo+emisGlo, emisFup+emisGup, color=cmap(6), alpha=0.15)
  ax1.legend(loc=0)

  ax2.loglog(Telist, emisE/(emisF+emisG), color=cmap(0), label='Exp')
  ax2.loglog(Telist, emisEUrd/(emisFUrd+emisGUrd), linestyle='--', color=cmap(1), label='AtomDB')
  ax2.fill_between(Telist, emisElo/(emisFup+emisGup), emisEup/(emisFlo+emisGlo), color=cmap(0), alpha=0.15) #error bars are using low-H/high-He or vice versa
  ax2.legend(loc=0)

  ax3.loglog(Telist, (emisE/(emisF+emisG))/(emisEUrd/(emisFUrd+emisGUrd)), label='H:He, Exp/AtomDB', color=cmap(0))
  ax3.fill_between((emisElo/(emisFlo+emisGlo))/(emisEUrd/(emisFUrd+emisGUrd)),(emisEup/(emisFup+emisGup))/(emisEUrd/(emisFUrd+emisGUrd)),color=cmap(0), alpha=0.15)
  ax3.legend(loc=0)

  ax1.set_xlim(2e6,4e8)
  ax1.set_ylim(1e-22, 1e-14)
  ax2.set_ylim(1e-4,1.1)
  ax3.set_ylim(1e-1, 1e1)
  ax1.set_ylabel('Emissivity (ph cm^3 s^-1)')
  ax2.set_ylabel('He/H-like')
  
  ax2.set_xlabel('Temperature (K)')