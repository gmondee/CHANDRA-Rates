import pyatomdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os

plt.ion()
plt.rcParams.update({'font.size': 16})
# example to get recombination rates and ionizatino balance

Z = 26
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
# fig = plt.figure()
#ax = fig.add_subplot(211, height_ratios=[2, 1])

chargesToHighlight=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,23,24,25,26]
fig, (ax, ax2) = plt.subplots(2,1, sharex=True, height_ratios=[2, 1])
for z1 in range(1, Z+2):
  ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(z1-1)
  if z1-1 in chargesToHighlight:
    chargeColor=cmap(z1-1)
    chargeLabel=ionsymb
    chargeWidthMod = 0.5
    chargezorder = 99
  else:
    chargeColor='black'
    chargeLabel=None
    chargeWidthMod = 0
    chargezorder = 2
  ax.loglog(Telist, ionbalUrdam[:,z1-1], linestyle='--', linewidth=1+chargeWidthMod, color=chargeColor, zorder=chargezorder)
  ax.loglog(Telist, ionbalExp[:,z1-1], linestyle='-', linewidth=1+chargeWidthMod, label=chargeLabel, color=chargeColor, zorder=chargezorder)
  ax.loglog(Telist, ionbalLower[:,z1-1], linestyle='-', linewidth=.5, color=chargeColor, zorder=chargezorder)
  ax.loglog(Telist, ionbalUpper[:,z1-1], linestyle='-', linewidth=.5, color=chargeColor, zorder=chargezorder)
  ax.fill_between(Telist, ionbalLower[:,z1-1], ionbalUpper[:,z1-1], color=chargeColor, alpha=0.15+chargeWidthMod/5, zorder=chargezorder)
  
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
ax.set_xlim(1e4, 1e9)
ax.set_ylim(4e-3,1.3)
CSLegend = ax.legend(loc='lower right', bbox_to_anchor=(1,.08), ncol=2, fontsize=10)
ExpUrdLegend = [matplotlib.lines.Line2D([], [], color='black', linestyle=ls[0], label=ls[1]) for ls in [['-','Experiment'],['--','Urdampilleta']]]
ax.add_artist(CSLegend)
leg = ax.legend(handles=ExpUrdLegend, loc='lower right', bbox_to_anchor=(1,0), ncol=2, fontsize=10)
leg.set_zorder(100)
CSLegend.set_zorder(100)
ax.set_ylabel('Ion Fraction')

lowerLim=8e-3
for charge in range(Z+1):
  if charge in chargesToHighlight:
    chargeColor=cmap(charge)
    chargeWidthMod = 0.5
    chargezorder = 99
  else:
    chargeColor='black'
    chargeWidthMod = 0
    chargezorder = 2
  ionsymb = pyatomdb.atomic.Ztoelsymb(Z)+'$^{%i+}$'%(charge)
  Slice = np.where(ionbalExp[:,charge]>lowerLim)
  ax2.loglog(Telist[Slice],ionbalExp[:,charge][Slice]/ionbalUrdam[:,charge][Slice], color=chargeColor, label=ionsymb, zorder=chargezorder)
  ax2.fill_between(Telist[Slice], ionbalLower[:,charge][Slice]/ionbalUrdam[:,charge][Slice], ionbalUpper[:,charge][Slice]/ionbalUrdam[:,charge][Slice],
                  color=chargeColor, alpha=0.09+chargeWidthMod/5, zorder=chargezorder)
ax2.set_yscale('linear')
ax2.set_xlim(1e4, 1e9)
ax2.set_ylim(0,3)
ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('Experiment/Urdampilleta')
ax2.grid(axis='y')
fig.tight_layout()

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(f"{element}IonbalHighlight{"".join([str(item) for item in chargesToHighlight])}.pdf")
plt.tight_layout()
pdf.savefig( fig )
pdf.close()
