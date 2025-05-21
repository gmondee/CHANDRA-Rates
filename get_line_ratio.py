import pyatomdb
import numpy
import matplotlib.pyplot as plt

plt.ion()
# The results of this look very flat, as the main thing that changes with temperature is meant to be the ionization
# balance, and I have essentially frozen that. 

Telist = numpy.logspace(4,9,21) # big list o' temperatures


Z = 8 # element
z1 = 7 # ion
up = 7 # upper level of resonance line
lo = 1 # lower level of resonance line

n  = pyatomdb.spectrum.NEISession(elements=[8])

taulist = 1e12 #(irrelevant)

# define initial population here
init_pop = {}
init_pop[Z] = numpy.zeros(Z+1)

# define the ionization population here, probably with the output of your code. This is an example.
init_pop[Z][Z-3]=0.1 # Li-like
init_pop[Z][Z-2]=0.6 # He-like
init_pop[Z][Z-1]=0.3 # H-like  


# freeze_ion_pop makes it calculate the emissivity based on the population you provide, as opposed to anything else.

e = n.return_line_emissivity(Telist, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 


# now change the ion charge to 8 and get the H-like emissivity. Need to sum 2 lines.
z1 = 8 # ion
up = 3 # upper level of resonance line
f = n.return_line_emissivity(Telist, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 

up = 4 # upper level of resonance line
g = n.return_line_emissivity(Telist, taulist, Z, z1, up, lo, init_pop=init_pop, freeze_ion_pop=True, teunit='K') 

# let's plot some graphs

fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)

ax1.loglog(Telist, e['epsilon'], label='He-like')
ax1.loglog(Telist, f['epsilon']+g['epsilon'], label='H-like')


ax2.loglog(Telist, e['epsilon']/(f['epsilon']+g['epsilon']))

ax1.legend(loc=0)
ax1.set_ylabel('Emissivity (ph cm^3 s^-1)')
ax2.set_ylabel('He/H-like')
 
ax2.set_xlabel('Temperature (K)')