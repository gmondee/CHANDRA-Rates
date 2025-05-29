import pyatomdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os

plt.ion()
plt.rcParams.update({'font.size': 16})
# example to get recombination rates and ionizatino balance

Z = 8
element = pyatomdb.atomic.Ztoelsymb(Z)

bal_picklename = f'{element}_bal.pickle'
### Pickle looks like {'ionbal':ionbalExp[:,charge],'upper':ionbalUpper[:,z1-1],'lower':ionbalLower[:,z1-1]}
with open(os.path.abspath(os.path.join('pickle',bal_picklename)), 'rb') as file:
  ionbal = pickle.load(file)

Telist = ionbal['TlistK'] #numpy.logspace(4,9,21) # big list o' temperatures
ionbalExp=np.zeros((len(Telist), Z+1), dtype=float)
ionbalUpper=np.zeros((len(Telist), Z+1), dtype=float)
ionbalLower=np.zeros((len(Telist), Z+1), dtype=float)

for charge in range(Z+1):
  ionbalExp[:,charge] = ionbal[charge]['ionbal']
  ionbalUpper[:,charge]=ionbal[charge]['upper']
  ionbalLower[:,charge]=ionbal[charge]['lower']



ionratesUrdam = np.zeros((len(Telist), Z), dtype=float) #[[Z0+, Z1+, Z2+,...], [Z0+, Z1+, Z2+...]]
recrates = np.zeros((len(Telist), Z), dtype=float)
datacache={}

for z1 in range(1,Z+1):
  itmp, rtmp = pyatomdb.atomdb.get_ionrec_rate(Telist, Te_unit='K', Z=Z,z1=z1, extrap=True, datacache=datacache)
  
  ionratesUrdam[:,z1-1]=itmp
  recrates[:,z1-1]=rtmp

ionbalUrdam = np.zeros((len(Telist), Z+1), dtype=float)
for iT in range(len(Telist)):
  ionbalUrdam[iT,:] = pyatomdb.apec.solve_ionbal( ionratesUrdam[iT,:], recrates[iT,:])





# Plot some results
cmap = plt.get_cmap("hsv", Z+1)

### without filled in errors
# fig = plt.figure()
# ax = fig.add_subplot(111)

# for z1 in range(1, Z+2):
#   ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(z1-1)
#   ax.loglog(Telist, ionbalUrdam[:,z1-1], linestyle='--', label=f'{ionsymb} Urd', color=cmap(z1-1))
#   ax.loglog(Telist, ionbalExp[:,z1-1], linestyle='-', label=f'{ionsymb} Exp', color=cmap(z1-1))
  
# ax.set_ylim(1e-3,1.3)
# ax.legend(loc=0, ncol=2)
# ax.set_xlabel('Temperature (K)')
# ax.set_ylabel('Ion Fraction')

### plot with errors
fig = plt.figure()
ax = fig.add_subplot(111)
for z1 in range(1, Z+2):
  ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(z1-1)
  ax.loglog(Telist, ionbalUrdam[:,z1-1], linestyle='--', linewidth=1, color=cmap(z1-1))
  ax.loglog(Telist, ionbalExp[:,z1-1], linestyle='-', linewidth=1, label=f'{ionsymb}', color=cmap(z1-1))
  ax.loglog(Telist, ionbalLower[:,z1-1], linestyle='-', linewidth=.5, color=cmap(z1-1))
  ax.loglog(Telist, ionbalUpper[:,z1-1], linestyle='-', linewidth=.5, color=cmap(z1-1))
  plt.fill_between(Telist, ionbalLower[:,z1-1], ionbalUpper[:,z1-1], color=cmap(z1-1), alpha=0.15)
  
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
ax.set_xlim(1e4, 3.4e7)
ax.set_ylim(4e-3,1.3)
CSLegend = ax.legend(loc='center right', bbox_to_anchor=(1,0.7), ncol=2, fontsize=12)
ExpUrdLegend = [matplotlib.lines.Line2D([], [], color='black', linestyle=ls[0], label=ls[1]) for ls in [['-','Experiment'],['--','Urdampilleta']]]
plt.gca().add_artist(CSLegend)
ax.legend(handles=ExpUrdLegend, loc='center right', bbox_to_anchor=(1,0.83), ncol=2, fontsize=12)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Ion Fraction')

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(f"{element}Ionbal.pdf")
#for fig in range(1, plt.gcf().number + 1):

plt.tight_layout()
pdf.savefig( fig )
pdf.close()