import numpy#, pyatomdb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.special
from scipy.optimize import curve_fit
from scipy.stats import norm
import astropy.io.fits as pyfits
from lmfit import minimize, Parameters, Parameter, report_fit
import urdam
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

## A lot of the stuff above might not be needed, sorry. I've copied and pasted.

# Here I define several functions for cross sections and rate coefficients.

# Younger formula has 5 coefficients:
#    ionization potential of shell (I) in eV
#    fit coefficients A, B, C, D, in 10-14 eV^2 cm^2
# Note that fit coefficient C is the Bethe limit, and can actually be fixed in most fits if there are very few data points.
# Energies and temperatures are in eV

# Urdampilleta formula has 6 coefficients:
#    ionization potential of shell (I) in eV
#    fit coefficients A, B, C, D, F in 10-14 eV^2 cm^2
#    I have used F here instead of E from the paper to avoid confusion with energy
# Note that fit coefficient C is the Bethe limit, and can actually be fixed in most fits if there are very few data points.
# Energies and temperatures are in eV

def evaluate_crosssection_younger_di(E, Ain, Bin, Cin, Din, Iin, printme=False, removemin=False):

  A = Ain*1e-14
  B = Bin*1e-14
  C = Cin*1e-14
  D = Din*1e-14
  I = Iin * 1.0 # sticking with eV

  u=E/I

  s = 1/(u*I**2) * ( A*(1-(1/u)) + \
                    B*(1-(1/u))**2 +\
                    C*numpy.log(u) + \
                    D* numpy.log(u)/u)

  # debugging options
  if printme:
    print("A: %e"%(A*(1-(1/u))))
    print("B: %e"%(B*(1-(1/u))**2))
    print("C: %e"%(C*numpy.log(u)))
    print("D: %e"%(D*numpy.log(u)/u))

  if removemin:
    s[E<I] = 0.0
  return s

def evaluate_crosssection_urdam_di(E, Ain, Bin, Cin, Din, Fin, Iin, printme=False, removemin=False):

  A = Ain*1e-14
  B = Bin*1e-14
  C = Cin*1e-14
  D = Din*1e-14
  F = Fin*1e-14
  I = Iin * 1.0 # stick with eV

  u=E/I

  mec2 = pyatomdb.const.ME_KEV*1000

  eps = E/mec2
  tau = I/mec2

  # 2 different ways of calculating R, the relativistic correction. The second is an approximation. Should give the same answer where we care.

  R= ((tau+2)/(eps+2)) * ( (eps+1)/(tau+1))**2 * ( ( (tau+eps) * (eps+2) * ((tau+1)**2))/ ( eps*(eps+2)*((tau+1)**2) + tau*(tau+2)))**1.5
  R= 1+1.5*eps +0.25*eps**2

  s = 1/(u*I**2) * ( A*(1-(1/u)) + \
                    B*(1-(1/u))**2 +\
                    C*R*numpy.log(u) + \
                    D* numpy.log(u)/numpy.sqrt(u) + \
                    F * numpy.log(u)/u)


  if printme:
    print("A: %e"%(A*(1-(1/u))))
    print("B: %e"%(B*(1-(1/u))**2))
    print("C: %e"%(C*numpy.log(u)))
    print("D: %e"%(D*numpy.log(u)/numpy.sqrt(u)))
    print("F: %e"%(F*numpy.log(u)/u))

  if removemin:
    s[E<I] = 0.0
  return s

# def evaluate_crosssection_di_Cfrozen(E, Ain, Bin, Din, printme=False, removemin=False):

  # A = Ain*1e-14
  # B = Bin*1e-14
  # C = C_Frozen*1e-14
  # D = Din*1e-14

  # u=E/I

  # s = 1/(u*I**2) * ( A*(1-(1/u)) + \
                    # B*(1-(1/u))**2 +\
                    # C*numpy.log(u) + \
                    # D* numpy.log(u)/u)


  # if printme:
    # print("A: %e"%(A*(1-(1/u))))
    # print("B: %e"%(B*(1-(1/u))**2))
    # print("C: %e"%(C*numpy.log(u)))
    # print("D: %e"%(D*numpy.log(u)/u))

  # if removemin:
    # s[E<I] = 0.0
  # return s

# def evaluate_crosssection_di_mewe_Cfrozen(E, Ain, Bin, Din, Fin, I, printme=False, removemin=False):
  # s=evaluate_crosssection_di_mewe(E, Ain, Bin, C_Frozen, Din, Fin, I, printme=printme, removemin=removemin)

  # return s

def evaluate_crosssection_urdam_di_lm(params, E, data):
  # because of the way the fit routine works, have to evaluate then subtract from the observed data
  s=evaluate_crosssection_urdam_di(E, params['A'].value, params['B'].value, params['C'].value, params['D'].value, params['F'].value, params['I'].value)
  return s-data


def evaluate_upsilon_urdam_di(kT, Ain, Bin, Cin, Din, Fin, Iin, printme=False, removemin=False):

  # This is a code from another source, so it requires a different unit conversion
  #
  # Input, coefficients are the same as in the other routines, no need to convert before feeding them in here.

  A = Ain*1e-24
  B = Bin*1e-24
  C = Cin*1e-24
  D = Din*1e-24
  F = Fin*1e-24

  coeff = {}
  coefft={}
  coefft['I_keV'] = I/1000
  coefft['A'] = A
  coefft['B'] = B
  coefft['C'] = C
  coefft['D'] = D
  coefft['E'] = F
  coeff[0] = coefft
  ups = urdam.eval_udi(coeff, kT)

  return ups


class dataset():
  """
  This is probably overkill: designed to hold all the relevant info on a dataset, including plot styles as I was trying to match things up between several plots.
  """


  def __init__(self, energy, sigma, name, color, marker, markersize, linestyle, fillstyle, sigma_error = None, ionpot = 8828.0):
    self.energy_in=energy
    self.sigma_in = sigma
    self.name = name
    self.color=color
    self.marker=marker
    self.markersize=markersize
    self.linestyle=linestyle
    self.fillstyle=fillstyle
    self.ionpot=ionpot

    self.sigma_error_in = sigma_error

  def set_p0(self, p0):
    self.p0 = p0

  def set_lm_params(self, p0, minval=-1e3, maxval=1e3):
    # Initialize the fit parameters.
    # To freeze them (e.g. C or I), do p.params['C'].vary=False
    # p0 is an initial guess at the parameters

    self.params = Parameters()

    self.params.add('A', value= p0[0], min=minval, max=maxval)
    self.params.add('B', value= p0[1], min=minval, max=maxval)
    self.params.add('C', value= p0[2], min=minval, max=maxval)
    self.params.add('D', value= p0[3], min=minval, max=100)
    self.params.add('F', value= p0[4], min=minval, max=maxval)
    self.params.add('I', value= p0[5], min=8600, max=8900)


  def calc_urdam_coeffts_lm(self):

    result = minimize(evaluate_crosssection_urdam_di_lm, self.params, args=(self.energy_in, self.sigma_in))
    report_fit(self.params)

    coeffts = numpy.array([result.params['A'].value, result.params['B'].value, result.params['C'].value, result.params['D'].value, result.params['F'].value, result.params['I'].value])

    self.coeffts=coeffts


  def evaluate_crosssection(self, energy):
    """
    energy in eV
    """
    s=evaluate_crosssection_urdam_di(energy,self.coeffts[0],\
                                            self.coeffts[1],\
                                            self.coeffts[2],\
                                            self.coeffts[3],\
                                            self.coeffts[4],\
                                            self.coeffts[5])

    return(s)

  def evaluate_upsilon(self, kT):
    """
    kT in eV
    """

    coeffts = self.coeffts
    ups = evaluate_upsilon_urdam_di(kT, \
                                    self.coeffts[0], \
                                    self.coeffts[1], \
                                    self.coeffts[2], \
                                    self.coeffts[3], \
                                    self.coeffts[4], \
                                    self.coeffts[5])
    return ups





# EXAMPLE DATA SET

# these are the initial guesses fit coefficients (from literature) in keV^2 m^2
p0_mewe = numpy.array([3.43699989e-23, -1.40600004e-23, 3.77999997e-24, 0., -2.91000004e-23])
I = 8828.0 # ionization potential in eV
# convert into 1e-24 keV^2 m^2 (or 1e-14 eV^2 cm^2)
p0_mewe *= 1e24
p0_mewe = numpy.append(p0_mewe, I) # add on the ionization potential in eV.

# Data from experiment
e_in_yang = numpy.array([ 9100, 9390, 9890, 11900, 17920.0, 52968,  176560]) #energy in eV
sig_out_yang = numpy.array([2.17256081948923E-23, 5.00098759638621E-23, 8.62906294287617E-23, 2.3931696519681E-22, 4.95317213025083E-22,3.866033998101439e-22,1.7279646243769038e-22  ]) #cross sections in cm^2
sig_out_err_yang = numpy.array([1.57438246606618E-24, 3.66284073809203E-24, 6.4259610113903E-24, 1.89624622318014E-23, 4.15264225830204E-23, 3.866033998101439e-23,1.7279646243769038e-23 ]) # uncertainties in cm^2

# getting a little elaborate here, can store multiple datasets
datasets= []

# set up output energy and temperature grids
kTlist = numpy.logspace(0,5,1251) # eV

Tlist = kTlist*11604.5 # in K
Elist = numpy.logspace(numpy.log10(I),6.4,1001) # in eV




























# This is the bit where I make 1000 copeis of each run. You probably don't need this.
# sigma_list_yang = numpy.zeros([len(e_in_yang), nruns])

# for i_s in range(len(sig_out_yang)):
  # sl = norm.rvs(loc=sig_out_yang[i_s], scale = sig_out_err_yang[i_s], size=nruns)
  # sigma_list_yang[i_s, :] = sl

# store_sigma_yang = numpy.zeros([nruns, len(Elist)])
# store_upsilon_yang = numpy.zeros([nruns, len(kTlist)])



# tmpdatasets = []
# for i in range(nruns):
  # t=dataset(e_in_yang, sigma_list_yang[:,i], 'Iter%i'%(i), '#0000', 'x', 5, '-', 'full')
  # yang_coeffts_tmp, yang_sigma_tmp, yang_upsilon_tmp = fit_coeffts(e_in_yang, sigma_list_yang[:,i], p0, ionpot, C_Frozen)
  # t.set_lm_params(p0_mewe)
  # t.calc_mewe_coeffts_lm(cfrozen=True)
  # stmp = t.evaluate_crosssection(Elist, mewe=True)
  # utmp = t.evaluate_upsilon(kTlist/1000)

  # store_sigma_yang[i,: ] = stmp
  # store_upsilon_yang[i,: ] = utmp


# iup = round(nruns/2 + (0.691/2*nruns))
# ilo = round(nruns/2 - (0.691/2*nruns))

# sigma_uplim_yang = numpy.zeros(len(Elist))
# sigma_lolim_yang = numpy.zeros(len(Elist))
# for i in range(len(Elist)):
  # s=numpy.sort(store_sigma_yang[:,i])
  # sigma_uplim_yang[i] = s[iup]
  # sigma_lolim_yang[i] = s[ilo]

# upsilon_uplim_yang = numpy.zeros(len(kTlist))
# upsilon_lolim_yang = numpy.zeros(len(kTlist))
# for i in range(len(kTlist)):
  # s=numpy.sort(store_upsilon_yang[:,i])
  # upsilon_uplim_yang[i] = s[iup]
  # upsilon_lolim_yang[i] = s[ilo]



#limit_linestyle_yang, = ax.loglog(Elist, sigma_uplim_yang, ':k', label=r'$1\sigma$')
#ax.loglog(Elist, sigma_lolim_yang, color=limit_linestyle_yang.get_color(), linestyle=limit_linestyle_yang.get_linestyle())




MARKERSIZE=4
LINEWIDTH=1.0
#datasets.append(dataset(e_in_fang, sig_out_fang, 'DWBE', '#6700CC', 'o', MARKERSIZE, '-', 'full'))
#datasets.append(dataset(e_in_younger_calc, sig_out_younger_calc, 'DWEA', '#ffbf40','^', MARKERSIZE, '-', 'full'))
#datasets.append(dataset(e_in_fursa_frdwa, sig_out_fursa_frdwa, 'RDW', '#0000FF', 's', MARKERSIZE, '-', 'none'))
#datasets.append(dataset(e_in_fursa_ccc, sig_out_fursa_ccc, 'CCC', '#888888', '*', MARKERSIZE, '-.', 'full', ionpot=8758.05))
#datasets.append(dataset(e_in_fursa_dwa, sig_out_fursa_dwa, 'DWBA', '#ff40d9', '2', MARKERSIZE, '--', 'full'))
#datasets.append(dataset(e_in_fursa_rccc, sig_out_fursa_rccc, 'RCCC', '#ff0000', 'o', MARKERSIZE, '-', 'none'))
#datasets.append(dataset(e_in_yurifac, sig_out_yurifac, 'FAC_DW', '#00CC00', 'd', MARKERSIZE, '-', 'full'))
datasets.append(dataset(e_in_yang, sig_out_yang, 'This Work', '#000000', 's', MARKERSIZE, '-', 'none', sigma_error=sig_out_err_yang))
#datasets.append(dataset(e_in_yang, sigma_uplim_yang, '1-sigma', '#000000', marker='o', linestyle=':', fillstyle='full'))
#datasets.append(dataset(e_in_yang, sigma_lolim_yang, '1-sigma', '#000000', marker='o', linestyle=':', fillstyle='full'))

#edatasets=[]
#edatasets.append(dataset(Tlist, sigma_uplim_yang, '1-sigma', '#000000', 'none',MARKERSIZE, ':', 'none'))
#edatasets.append(dataset(Tlist, sigma_lolim_yang, '1-sigma', '#000000', 'none',MARKERSIZE, ':', 'none'))


fig5 = plt.figure(figsize=(6.5,4.5))

gs = GridSpec(3, 2, figure=fig5)
ax5 = fig5.add_subplot(gs[:2, 0])
ax6 = fig5.add_subplot(gs[2, 0], sharex=ax5)
ax7 = fig5.add_subplot(gs[:2, 1])
ax8 = fig5.add_subplot(gs[2, 1], sharex=ax7)

fig6 = plt.figure(figsize=(6.5,4.5))

gs2 = GridSpec(3, 1, figure=fig6)
ax9 = fig6.add_subplot(gs2[:2, 0])
ax10 = fig6.add_subplot(gs2[2, 0], sharex=ax9)


fig5.show()
fig5.subplots_adjust(left=0.105,right=0.97, top=0.98, bottom=0.124, hspace=0)
fig6.show()
fig6.subplots_adjust(left=0.12,right=0.97, top=0.98, bottom=0.124, hspace=0.01)
#print(p0, p0_mewe)
for d in datasets:

  # set the initial parameters
  d.set_lm_params(p0_mewe)

  # freeze the ionization potential and C (optional!)
  d.params['C'].vary=False
  d.params['I'].vary=False

  # calculate the fit coefficients
  d.calc_urdam_coeffts_lm()
  print("Coefficients for %s:"%(d.name), d.coeffts)


  # All of this is plotting now: get the cross sections from the fitted model, plot them against the real data
  stmp = d.evaluate_crosssection(Elist)
  ax5.loglog(Elist/1e3, stmp*1e22, color=d.color, linestyle=d.linestyle, linewidth=LINEWIDTH)
  ax7.semilogy(Elist/1e3, stmp*1e22, color=d.color, linestyle=d.linestyle, linewidth=LINEWIDTH)
  stmp = d.evaluate_crosssection(d.energy_in)
  ax6.loglog(d.energy_in/1e3, stmp/d.sigma_in, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle, color=d.color, linestyle='none', linewidth=LINEWIDTH)
  ax8.semilogy(d.energy_in/1e3, stmp/d.sigma_in, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle, color=d.color, linestyle='none', linewidth=LINEWIDTH)

#  ax7.plot(d.energy_in/1e3, d.sigma_in*1e22, marker=d.marker, fillstyle=d.fillstyle, color=d.color, label=d.name, linestyle='none')
  # If the cross section data has error bars, use it.
  if d.sigma_error_in is not None:
    ax5.errorbar(d.energy_in/1e3, d.sigma_in*1e22, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle,color=d.color, label=d.name,capsize=2, yerr=d.sigma_error_in*1e22, linestyle='none', zorder=99)
    ax7.errorbar(d.energy_in/1e3, d.sigma_in*1e22, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle,color=d.color, label=d.name,capsize=2, yerr=d.sigma_error_in*1e22, linestyle='none', zorder=99)
    ax6.errorbar(d.energy_in/1e3, stmp/d.sigma_in, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle,color=d.color, label=d.name,capsize=2, yerr=d.sigma_error_in/stmp, linestyle='none', zorder=99)
    ax8.errorbar(d.energy_in/1e3, stmp/d.sigma_in, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle,color=d.color, label=d.name,capsize=2, yerr=d.sigma_error_in/stmp, linestyle='none', zorder=99)

  else:
    ax5.plot(d.energy_in/1e3, d.sigma_in*1e22, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle, color=d.color, label=d.name, linestyle='none', linewidth=LINEWIDTH)
    ax7.plot(d.energy_in/1e3, d.sigma_in*1e22, marker=d.marker, markersize=d.markersize, fillstyle=d.fillstyle, color=d.color, label=d.name, linestyle='none', linewidth=LINEWIDTH)

  ytmp = d.evaluate_upsilon(kTlist/1000)
  ax9.loglog(Tlist, ytmp,    color=d.color, linestyle=d.linestyle, label=d.name, linewidth=LINEWIDTH)# plotting of uncertainties, skip!
#  ax10.loglog(Tlist, ytmp/uups,    color=d.color, linestyle=d.linestyle, linewidth=LINEWIDTH)#ax5.plot(Elist/1e3, sigma_uplim_yang*1e22, color='k', linestyle=':', linewidth=LINEWIDTH)
#ax7.plot(Elist/1e3, sigma_uplim_yang*1e22, color='k', linestyle=':', linewidth=LINEWIDTH)
#ax6.plot(Elist/1e3, sigma_uplim_yang/datasets[-1].evaluate_crosssection_urdam_di(Elist), color='k', linestyle=':', linewidth=LINEWIDTH)
#ax8.plot(Elist/1e3, sigma_uplim_yang/datasets[-1].evaluate_crosssection_urdam_di(Elist), color='k', linestyle=':', linewidth=LINEWIDTH)

#ax5.plot(Elist/1e3, sigma_lolim_yang*1e22, color='k', linestyle=':', linewidth=LINEWIDTH)
#ax7.plot(Elist/1e3, sigma_lolim_yang*1e22, color='k', linestyle=':', linewidth=LINEWIDTH)
#ax6.plot(Elist/1e3, sigma_lolim_yang/datasets[-1].evaluate_crosssection_urdam_di(Elist), color='k', linestyle=':', linewidth=LINEWIDTH)
#ax8.plot(Elist/1e3, sigma_lolim_yang/datasets[-1].evaluate_crosssection_urdam_di(Elist), color='k', linestyle=':', linewidth=LINEWIDTH)

#urdamcoefft=[8.82800007*1e3, 3.43699989e-23*1e24, -1.40600004e-23*1e24, 3.77999997e-24*1e24, 0.*1e24, -2.91000004e-23*1e24]
#ax6.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax8.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax6.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
#ax8.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
#uups = evaluate_upsilon_di_mewe(kTlist/1000, urdamcoefft[0], \
#                                       urdamcoefft[1], \
#                                       urdamcoefft[2], \
#                                       urdamcoefft[3], \
#                                       urdamcoefft[4], \
#                                       urdamcoefft[5])
#
#for d in datasets:
#
#  ytmp = d.evaluate_upsilon(kTlist/1000)
#  ax9.loglog(Tlist, ytmp,    color=d.color, linestyle=d.linestyle, label=d.name, linewidth=LINEWIDTH)
#  ax10.loglog(Tlist, ytmp/uups,    color=d.color, linestyle=d.linestyle, linewidth=LINEWIDTH)

#ax9.plot(Tlist, upsilon_uplim_yang, color='k', linestyle=':', label='1$\sigma$', linewidth=LINEWIDTH)
#ax9.plot(Tlist, upsilon_lolim_yang, color='k', linestyle=':', linewidth=LINEWIDTH)
#ax10.plot(Tlist, upsilon_uplim_yang/uups, color='k', linestyle=':', label='1$\sigma$', linewidth=LINEWIDTH)
#ax10.plot(Tlist, upsilon_lolim_yang/uups, color='k', linestyle=':', linewidth=LINEWIDTH)
#ax10.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax10.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

#uline, = ax9.loglog(Tlist, uups, label='Urdampilleta')
plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax7.get_xticklabels(), visible=False)
plt.setp(ax9.get_xticklabels(), visible=False)
ax5.set_xlim(8.870,1e2)
ax7.set_xlim(8.870,12.2)
ax5.set_ylim(0.05,7)
ax7.set_ylim(0.1,3)
ax6.set_ylim(0.6,1.3)
ax8.set_ylim(0.6,1.3)

ax9.set_ylim(1e-20,1e-11)
ax9.set_xlim(5e6, 1e9)
ax10.set_ylim(0.6,1.3)

ax6.set_xlabel("Electron Energy (keV)")
ax8.set_xlabel("Electron Energy (keV)")
ax5.set_ylabel("EII Ionization Cross Section ( $10^{-22}$cm$^2$ )")
ax6.set_ylabel("Ratio fit/data")

ax10.set_xlabel("Electron Temperature (K)")
ax9.set_ylabel("Ionization Rate Coefft (cm$^3$s$^{-1}$ )")
ax10.set_ylabel("Ratio to Urdampilleta")

ax6.axhline(1.0, color='black', linestyle=':')
ax5.legend(ncol=2)
ax9.legend(ncol=2)
ax5.grid(True, which='both')
ax6.grid(True, which='both')
ax7.grid(True, which='both')
ax8.grid(True, which='both')
ax9.grid(True, which='both')
ax10.grid(True, which='both')

iondat = pyfits.open('/export1/atomdb_latest/APED/ionbal/eigen/eigenfe_v3.0.7.fits')
T = numpy.logspace(4,9,1251)
feqb = numpy.array(iondat[1].data['FEQB'])

Tlo = T[feqb[:,24]>0.1]
Thi = T[feqb[:,24]>0.3]
c = 'orange'

ax9.plot([Tlo[0],Thi[0]], [6e-12,6e-12], ':', color=c, linewidth=3)
ax9.plot([Tlo[-1],Thi[-1]], [6e-12,6e-12], ':', color=c, linewidth=3)
ax9.plot([Thi[0],Thi[-1]], [6e-12,6e-12], '-', color=c, linewidth=3)

Tlo = T[feqb[:,25]>0.1]
Thi = T[feqb[:,25]>0.3]
c = 'brown'

ax9.plot([Tlo[0],Thi[0]], [8e-12,8e-12], ':', color=c, linewidth=3)
ax9.plot([Tlo[-1],Thi[-1]], [8e-12,8e-12], ':', color=c, linewidth=3)
ax9.plot([Thi[0],Thi[-1]], [8e-12,8e-12], '-', color=c, linewidth=3)



#ax5.xaxis.grid(True)
#ax6.xaxis.grid(True)
plt.draw()
zzz=input('hola')

for ext in ['pdf','svg','png']:
  fig5.savefig("xsec_fit.%s"%(ext))
  fig6.savefig("ratecoeff_fit.%s"%(ext))
### NOW LETS FIX THIS

# GET THE NEW IONBAL
fig7= plt.figure(figsize=(6.5,4))
fig7.show()
ax1 = fig7.add_subplot(211)
ax2 = fig7.add_subplot(212, sharex=ax1)

#T=numpy.logspace(6.5,9,201)
ionrates = numpy.zeros((len(Tlist),26), dtype=float)
recrates = numpy.zeros((len(Tlist),26), dtype=float)
ionpop = numpy.zeros((len(Tlist), 27), dtype=float)
for z1 in range(1,27):
  i,r = pyatomdb.atomdb.get_ionrec_rate(Tlist, Z=26, z1=z1)
  ionrates[:,z1-1]=i
  recrates[:,z1-1]=r

for i in range(len(T)):
  pop = pyatomdb.apec.solve_ionbal(ionrates[i,:], recrates[i,:])
  ionpop[i,:] = pop

ionrates2 = ionrates.copy()
ionpop2 = numpy.zeros((len(Tlist), 27), dtype=float)

ytmp = datasets[-1].evaluate_upsilon(Tlist/(11604.5*1000))

ionrates2[:,24]=ytmp
for i in range(len(T)):
  pop = pyatomdb.apec.solve_ionbal(ionrates2[i,:], recrates[i,:])
  ionpop2[i,:] = pop


ionrates2up = ionrates2.copy()
ionpop2up = numpy.zeros((len(Tlist), 27), dtype=float)

#upsilon_uplim_yang
ytmp = upsilon_uplim_yang
print(ionrates2up[:,24])
print(ytmp)

ionrates2up[:,24]=ytmp
for i in range(len(T)):
  pop = pyatomdb.apec.solve_ionbal(ionrates2up[i,:], recrates[i,:])
  ionpop2up[i,:] = pop

ionrates2lo = ionrates2.copy()
ionpop2lo = numpy.zeros((len(Tlist), 27), dtype=float)

ytmp = upsilon_lolim_yang

print(ionrates2lo[:,24])
print(ytmp)

ionrates2lo[:,24]=ytmp
for i in range(len(T)):
  pop = pyatomdb.apec.solve_ionbal(ionrates2lo[i,:], recrates[i,:])
  ionpop2lo[i,:] = pop


fe24,=ax1.loglog(T, ionpop2[:,24])
fe25,=ax1.loglog(T, ionpop2[:,25])


ax1.loglog(T, ionpop[:,24], color=fe24.get_color(), linestyle='-.')
ax1.loglog(T, ionpop[:,25], color=fe25.get_color(), linestyle='-.')

ax1.loglog(T, ionpop2up[:,24], color=fe24.get_color(), linestyle=':')
ax1.loglog(T, ionpop2up[:,25], color=fe25.get_color(), linestyle=':')
ax1.loglog(T, ionpop2lo[:,24], color=fe24.get_color(), linestyle=':')
ax1.loglog(T, ionpop2lo[:,25], color=fe25.get_color(), linestyle=':')


ax2.loglog(T, ionpop2[:,24]/ionpop[:,24], color=fe24.get_color())
ax2.loglog(T, ionpop2[:,25]/ionpop[:,25], color=fe25.get_color())

ax2.loglog(T, ionpop2up[:,24]/ionpop[:,24], color=fe24.get_color(), linestyle=':')
ax2.loglog(T, ionpop2up[:,25]/ionpop[:,25], color=fe25.get_color(), linestyle=':')

ax2.loglog(T, ionpop2lo[:,24]/ionpop[:,24], color=fe24.get_color(), linestyle=':')
ax2.loglog(T, ionpop2lo[:,25]/ionpop[:,25], color=fe25.get_color(), linestyle=':')
ax1.set_ylim(0.01,1)
ax1.set_xlim(8e6,8e8)
ax2.set_ylim(0.75, 1.2)
fig7.subplots_adjust(left=0.124,right=0.98, top=0.98, bottom=0.133, hspace=0.0)
ax2.set_xlabel('Electron Temperature (K)')
ax1.set_ylabel('Fractional abund')
ax2.set_ylabel('ratio/Dere')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

plt.setp(ax1.get_xticklabels(), visible=False)

custom_lines = [fe24, fe25,Line2D([0], [0], color='#000000', linestyle='-.'),
                Line2D([0], [0], color='#000000', linestyle='-'),
                Line2D([0], [0], color='#000000', linestyle=':')]
labels=['Fe$^{24+}$', 'Fe$^{25+}$', 'Dere','This work','$1\sigma$']

ax1.legend(custom_lines, labels, loc=0)
ax1.grid(True, which='both')
ax2.grid(True, which='both')
plt.draw()
zzz=input()
for suff in ['png','svg','pdf']:
   fig7.savefig('ionfrac.%s'%(suff))


fig = plt.figure(figsize=(6.5,4))
fig.show()
fig.subplots_adjust(left=0.105,right=0.97, top=0.98, bottom=0.124)
fig2 = plt.figure(figsize=(3,5))
fig2.subplots_adjust(left=0.23,right=0.95, top=0.98, bottom=0.1, hspace=0.0)
fig2.show()
ax = fig.add_subplot(121)
#ax4 = ax.inset_axes(
#    [0.1, 0.1, 0.47, 0.47])
ax4 = fig.add_subplot(122, sharey=ax)

ax2 = fig2.add_subplot(211)
ax3 = fig2.add_subplot(212, sharex=ax2)


ax.errorbar(e_in_yang, sig_out_yang, color='k', capsize=2, yerr=sig_out_err_yang, linestyle='none',label=r'Measured', zorder=99)

ax4.errorbar(e_in_yang, sig_out_yang, color='k', capsize=2, yerr=sig_out_err_yang, linestyle='none',label=r'Measured', zorder=99)





#yangly1_coeffts, yangly1_sigma, yangly1_upsilon = fit_coeffts(e_in_yangly1, sig_out_yangly1, p0, ionpot, C_Frozen)
#yangly2_coeffts, yangly2_sigma, yangly2_upsilon = fit_coeffts(e_in_yangly2, sig_out_yangly2, p0, ionpot, C_Frozen)
yang_coeffts, yang_sigma, yang_upsilon = fit_coeffts(e_in_yang, sig_out_yang, p0, ionpot, C_Frozen)
yang_coeffts, yang_sigma, yang_upsilon = fit_coeffts(e_in_yang, sig_out_yang, p0_C, ionpot, C_Frozen, use_Cfrozen=False)

# repeat including high enregy point
#dipti_coeffts, dipti_sigma, dipti_upsilon = fit_coeffts(e_in_dipti[:-1], sig_out_dipti[:-1], p0, ionpot, C_Frozen)

# repeat including high enregy point
#diptifac_coeffts, diptifac_sigma, diptifac_upsilon = fit_coeffts(e_in_diptifac[:-1], sig_out_diptifac[:-1], p0, ionpot, C_Frozen)

# repeat including high enregy point
yurifac_coeffts, yurifac_sigma, yurifac_upsilon = fit_coeffts(e_in_yurifac, sig_out_yurifac, p0, ionpot, C_Frozen)
yurifac_coeffts, yurifac_sigma, yurifac_upsilon = fit_coeffts(e_in_yurifac, sig_out_yurifac, p0_C, ionpot, C_Frozen, use_Cfrozen=False)

# repeat including high enregy point
fursa_frdwa_coeffts, fursa_frdwa_sigma, fursa_frdwa_upsilon = fit_coeffts(e_in_fursa_frdwa, sig_out_fursa_frdwa, p0, ionpot, C_Frozen)

# repeat including high enregy point
fursa_dwa_coeffts, fursa_dwa_sigma, fursa_dwa_upsilon = fit_coeffts(e_in_fursa_dwa, sig_out_fursa_dwa, p0, ionpot, C_Frozen)

# repeat including high enregy point
fursa_ccc_coeffts, fursa_ccc_sigma, fursa_ccc_upsilon = fit_coeffts(e_in_fursa_ccc, sig_out_fursa_ccc, p0, ionpot, C_Frozen)

fursa_rccc_coeffts, fursa_rccc_sigma, fursa_rccc_upsilon = fit_coeffts(e_in_fursa_rccc, sig_out_fursa_rccc, p0, ionpot, C_Frozen)
fursa_rccc_coeffts, fursa_rccc_sigma, fursa_rccc_upsilon = fit_coeffts(e_in_fursa_rccc, sig_out_fursa_rccc, p0_C, ionpot, C_Frozen, use_Cfrozen=False)

fang_coeffts, fang_sigma, fang_upsilon = fit_coeffts(e_in_fang, sig_out_fang, p0, ionpot, C_Frozen)

younger_calc_coeffts, younger_calc_sigma, younger_calc_upsilon = fit_coeffts(e_in_younger_calc, sig_out_younger_calc, p0, ionpot, C_Frozen)

print("Coefficients calculated (I A B C D)")
print("Mazotta coeffts:                                               ",  maz_coeffts)
#print("Yang with fixed , fixed C, no high E point, no coefficient constraints:", yang_coeffts)
print("Yang from LyA1+2 , fixed C, no high E point, no coefficient constraints:", yang_coeffts)
#print("Dipti with fixed C, no high E point, no coefficient constraints:", dipti_coeffts)
#print("DiptiFAC with fixed C, no high E point, no coefficient constraints:", diptifac_coeffts)
print("YuriFAC with fixed C, no high E point, no coefficient constraints:", yurifac_coeffts)
print("Fursa FRDWA with fixed C, no high E point, no coefficient constraints:", fursa_frdwa_coeffts)
print("Fursa DWA with fixed C, no high E point, no coefficient constraints:", fursa_dwa_coeffts)
print("Fursa CCC with fixed C, no high E point, no coefficient constraints:", fursa_ccc_coeffts)
print("Fursa RCCC with fixed C, no high E point, no coefficient constraints:", fursa_rccc_coeffts)
print("Fang with fixed C, no high E point, no coefficient constraints:", fang_coeffts)
print("Younger with fixed C, no high E point, no coefficient constraints:", younger_calc_coeffts)


# plot stuff

ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Cross section (cm$^2$)')
ax3.set_xlabel('Temeprature (eV)')
ax2.set_ylabel('Rate Coefft (cm$^3$ s$^{-1}$)')



# get the Dere numbers from AtomDB
pyatomdb.util.switch_version('3.0.9')
atomdb_upsilon,zzz = pyatomdb.atomdb.get_ionrec_rate(kTlist/1000, Z=26, z1=25, Te_unit='keV')

# Use Younger formula coefficients
an = [27.0, -60.1, 140.0, -89.9]
bn = [-9.62, 33.1, -82.5, 54.6]
cn = [3.69, 4.32, -2.527, 0.262]
dn = [-21.7, 42.5, -131., 87.4]

A, B, C, D = calc_coeffts(an, bn, cn, dn, 26, 2)
younger_coeffts=numpy.array([I, A, B, C, D])
print("Younger coeffts: I=%f A=%f B=%f C=%f D=%f"%(I, A, B, C, D))
younger_sigma =  evaluate_crosssection_di(Elist, A, B, C, D)
younger_upsilon = pyatomdb.atomdb._ci_younger(Tlist, younger_coeffts)


# sigma_list_ly2 = numpy.zeros([len(e_in_yangly2), nruns])

# for i_s in range(len(sig_out_yangly2)):
  # sl = norm.rvs(loc=sig_out_yangly2[i_s], scale = sig_out_err_yangly2[i_s], size=nruns)
  # sigma_list_ly2[i_s, :] = sl

# store_sigma_ly2 = numpy.zeros([nruns, len(Elist)])
# store_upsilon_ly2 = numpy.zeros([nruns, len(kTlist)])


# for i in range(nruns):

  # yang_coeffts_tmp, yang_sigma_tmp, yang_upsilon_tmp = fit_coeffts(e_in_yangly2, sigma_list_ly2[:,i], p0, ionpot, C_Frozen)

  # store_sigma_ly2[i,: ] = yang_sigma_tmp
  # store_upsilon_ly2[i,: ] = yang_upsilon_tmp

# sigma_list_lyboth = numpy.zeros([len(e_in_yanglyboth), nruns])

# for i_s in range(len(sig_out_yanglyboth)):
  # sl = norm.rvs(loc=sig_out_yanglyboth[i_s], scale = sig_out_err_yanglyboth[i_s], size=nruns)
  # sigma_list_lyboth[i_s, :] = sl

# store_sigma_lyboth = numpy.zeros([nruns, len(Elist)])
# store_upsilon_lyboth = numpy.zeros([nruns, len(kTlist)])


# for i in range(nruns):

  # yang_coeffts_tmp, yang_sigma_tmp, yang_upsilon_tmp = fit_coeffts(e_in_yanglyboth, sigma_list_lyboth[:,i], p0, ionpot, C_Frozen)

  # store_sigma_lyboth[i,: ] = yang_sigma_tmp
  # store_upsilon_lyboth[i,: ] = yang_upsilon_tmp



#  ax.loglog(Elist[yang_sigma_tmp>0], yang_sigma_tmp[yang_sigma_tmp>0], color='k', alpha = 0.1)

#  ax2.loglog(kTlist, yang_upsilon_tmp, color='k', alpha = 0.1)


#yangly2_linestyle, = ax.loglog(Elist[yangly2_sigma>0], yangly2_sigma[yangly2_sigma>0], label='YangLy2')
#yanglyboth_linestyle, = ax.loglog(Elist[yanglyboth_sigma>0], yanglyboth_sigma[yanglyboth_sigma>0], label='YangLyboth')
#dipti_linestyle, = ax.loglog(Elist[dipti_sigma>0], dipti_sigma[dipti_sigma>0], label='Dipti')
fang_linestyle, = ax.loglog(Elist[fang_sigma>0], fang_sigma[fang_sigma>0], label='DWBE', color='#6700CC')
younger_calc_linestyle, = ax.loglog(Elist[younger_calc_sigma>0], younger_calc_sigma[younger_calc_sigma>0], label='DWEA', color='#ffbf40')
#diptifac_linestyle, = ax.loglog(Elist[diptifac_sigma>0], diptifac_sigma[diptifac_sigma>0], label='DiptiFAC')
fursa_frdwa_linestyle, = ax.loglog(Elist[fursa_frdwa_sigma>0], fursa_frdwa_sigma[fursa_frdwa_sigma>0], label='RDW', color='#0000FF')
fursa_ccc_linestyle, = ax.loglog(Elist[fursa_ccc_sigma>0], fursa_ccc_sigma[fursa_ccc_sigma>0], label='CCC', color='#888888')
fursa_dwa_linestyle, = ax.loglog(Elist[fursa_dwa_sigma>0], fursa_dwa_sigma[fursa_dwa_sigma>0], label='DWBA', color='#ff40D9')
fursa_rccc_linestyle, = ax.loglog(Elist[fursa_rccc_sigma>0], fursa_rccc_sigma[fursa_rccc_sigma>0], label='RCCC', color='#ff0000')
yurifac_linestyle, = ax.loglog(Elist[yurifac_sigma>0], yurifac_sigma[yurifac_sigma>0], label='FAC_DW', color='#00CC00')
yang_linestyle, = ax.loglog(Elist[yang_sigma>0], yang_sigma[yang_sigma>0], label='This Work', color='000000')


ax.plot(e_in_fang, sig_out_fang, color=fang_linestyle.get_color(), linestyle='none', marker='o')
ax.plot(e_in_younger_calc, sig_out_younger_calc, color=younger_calc_linestyle.get_color(), linestyle='none', marker='^')
ax.plot(e_in_fursa_frdwa, sig_out_fursa_frdwa, color=fursa_frdwa_linestyle.get_color(), linestyle='none', marker='s', fillstyle='none')
ax.plot(e_in_fursa_ccc, sig_out_fursa_ccc, color=fursa_ccc_linestyle.get_color(), linestyle='none', marker='*')
ax.plot(e_in_fursa_dwa, sig_out_fursa_dwa, color=fursa_dwa_linestyle.get_color(), linestyle='none', marker='2')
ax.plot(e_in_fursa_rccc, sig_out_fursa_rccc, color=fursa_rccc_linestyle.get_color(), linestyle='none', marker='o', fillstyle='none')
ax.plot(e_in_yurifac, sig_out_yurifac, color=yurifac_linestyle.get_color(), linestyle='none', marker='d')


# get the data
fang_comparison = evaluate_crosssection_di(e_in_fang, fang_coeffts[0],fang_coeffts[1],fang_coeffts[2],fang_coeffts[3] )
fang_stddev= numpy.std(fang_comparison - sig_out_fang)
print('fang:', fang_stddev)

younger_calc_comparison = evaluate_crosssection_di(e_in_younger_calc, younger_calc_coeffts[0],younger_calc_coeffts[1],younger_calc_coeffts[2],younger_calc_coeffts[3] )
younger_calc_stddev= numpy.std(younger_calc_comparison - sig_out_younger_calc)
print('younger_calc:', younger_calc_stddev)

fursa_frdwa_comparison = evaluate_crosssection_di(e_in_fursa_frdwa, fursa_frdwa_coeffts[0],fursa_frdwa_coeffts[1],fursa_frdwa_coeffts[2],fursa_frdwa_coeffts[3] )
fursa_frdwa_stddev= numpy.std(fursa_frdwa_comparison - sig_out_fursa_frdwa)
print('fursa_frdwa:', fursa_frdwa_stddev)

fursa_ccc_comparison = evaluate_crosssection_di(e_in_fursa_ccc, fursa_ccc_coeffts[0],fursa_ccc_coeffts[1],fursa_ccc_coeffts[2],fursa_ccc_coeffts[3] )
fursa_ccc_stddev= numpy.std(fursa_ccc_comparison - sig_out_fursa_ccc)
print('fursa_ccc:', fursa_ccc_stddev)

fursa_dwa_comparison = evaluate_crosssection_di(e_in_fursa_dwa, fursa_dwa_coeffts[0],fursa_dwa_coeffts[1],fursa_dwa_coeffts[2],fursa_dwa_coeffts[3] )
fursa_dwa_stddev= numpy.std(fursa_dwa_comparison - sig_out_fursa_dwa)
print('fursa_dwa:', fursa_dwa_stddev)

fursa_rccc_comparison = evaluate_crosssection_di(e_in_fursa_rccc, fursa_rccc_coeffts[0],fursa_rccc_coeffts[1],fursa_rccc_coeffts[2],fursa_rccc_coeffts[3] )
fursa_rccc_stddev= numpy.std(fursa_rccc_comparison - sig_out_fursa_rccc)
print('fursa_rccc:', fursa_rccc_stddev)

yurifac_comparison = evaluate_crosssection_di(e_in_yurifac, yurifac_coeffts[0],yurifac_coeffts[1],yurifac_coeffts[2],yurifac_coeffts[3] )
yurifac_stddev= numpy.std(yurifac_comparison - sig_out_yurifac)
print('yurifac:', yurifac_stddev)

yang_comparison = evaluate_crosssection_di(e_in_yang, yang_coeffts[0],yang_coeffts[1],yang_coeffts[2],yang_coeffts[3] )
yang_stddev= numpy.std(yang_comparison - sig_out_yang)
print('yang:', yang_stddev)

#ax.plot(e_in_fang, sig_out_fang, color=fang_linestyle.get_color(), linestyle='none', marker='o')



younger_nifs_E = numpy.array([9712, 13240, 17660, 44150, 70630, 88290, 132400, 176600, 264900, 441500, 662200, 882900])
younger_nifs_sigma = numpy.array([9.005E-23,   2.839E-22,   3.714E-22,   3.559E-22,   2.841E-22,   2.491E-22,   1.907E-22,   1.551E-22,   1.139E-22,   7.552E-23,   5.387E-23,   4.216E-23])

#ax.loglog(younger_nifs_E,younger_nifs_sigma, '-x', label='Younger_NIFS')

iup = round(nruns/2 + (0.691/2*nruns))
ilo = round(nruns/2 - (0.691/2*nruns))

sigma_uplim_yang = numpy.zeros(len(Elist))
sigma_lolim_yang = numpy.zeros(len(Elist))
for i in range(len(Elist)):
  s=numpy.sort(store_sigma_yang[:,i])
  sigma_uplim_yang[i] = s[iup]
  sigma_lolim_yang[i] = s[ilo]

upsilon_uplim_yang = numpy.zeros(len(kTlist))
upsilon_lolim_yang = numpy.zeros(len(kTlist))
for i in range(len(kTlist)):
  s=numpy.sort(store_upsilon_yang[:,i])
  upsilon_uplim_yang[i] = s[iup]
  upsilon_lolim_yang[i] = s[ilo]



limit_linestyle_yang, = ax.loglog(Elist, sigma_uplim_yang, ':k', label=r'$1\sigma$')
ax.loglog(Elist, sigma_lolim_yang, color=limit_linestyle_yang.get_color(), linestyle=limit_linestyle_yang.get_linestyle())

#sigma_uplim_ly2 = numpy.zeros(len(Elist))
#sigma_lolim_ly2 = numpy.zeros(len(Elist))
#for i in range(len(Elist)):
  #s=numpy.sort(store_sigma_ly2[:,i])
  #sigma_uplim_ly2[i] = s[iup]
  #sigma_lolim_ly2[i] = s[ilo]

#upsilon_uplim_ly2 = numpy.zeros(len(kTlist))
#upsilon_lolim_ly2 = numpy.zeros(len(kTlist))
#for i in range(len(kTlist)):
  #s=numpy.sort(store_upsilon_ly2[:,i])
  #upsilon_uplim_ly2[i] = s[iup]
  #upsilon_lolim_ly2[i] = s[ilo]



#limit_linestyle_ly2, = ax.loglog(Elist, sigma_uplim_ly2, 'k--', label='1-sigma')
#ax.loglog(Elist, sigma_lolim_ly2, color=limit_linestyle_ly2.get_color())



#sigma_uplim_lyboth = numpy.zeros(len(Elist))
#sigma_lolim_lyboth = numpy.zeros(len(Elist))
#for i in range(len(Elist)):
#  s=numpy.sort(store_sigma_lyboth[:,i])
#  sigma_uplim_lyboth[i] = s[iup]
#  sigma_lolim_lyboth[i] = s[ilo]
#
#upsilon_uplim_lyboth = numpy.zeros(len(kTlist))
#upsilon_lolim_lyboth = numpy.zeros(len(kTlist))
#for i in range(len(kTlist)):
#  s=numpy.sort(store_upsilon_lyboth[:,i])
#  upsilon_uplim_lyboth[i] = s[iup]
#  upsilon_lolim_lyboth[i] = s[ilo]



#limit_linestyle_lyboth, = ax.loglog(Elist, sigma_uplim_lyboth, 'k:', label='1-sigma')
#ax.loglog(Elist, sigma_lolim_lyboth, color=limit_linestyle_lyboth.get_color())






ax2.loglog(kTlist, atomdb_upsilon, label='Dere')
ax2.loglog(kTlist, yang_upsilon, color=yang_linestyle.get_color())
#ax2.loglog(kTlist, dipti_upsilon, color=dipti_linestyle.get_color())
ax2.loglog(kTlist, younger_calc_upsilon, color=younger_calc_linestyle.get_color())
#ax2.loglog(kTlist, diptifac_upsilon, color=diptifac_linestyle.get_color())
ax2.loglog(kTlist, yurifac_upsilon, color=yurifac_linestyle.get_color())
ax2.loglog(kTlist, fursa_frdwa_upsilon, color=fursa_frdwa_linestyle.get_color())
ax2.loglog(kTlist, fursa_dwa_upsilon, color=fursa_dwa_linestyle.get_color())
ax2.loglog(kTlist, fursa_ccc_upsilon, color=fursa_ccc_linestyle.get_color())
ax2.loglog(kTlist, fursa_rccc_upsilon, color=fursa_rccc_linestyle.get_color())
ax2.loglog(kTlist, fang_upsilon, color=fang_linestyle.get_color())
ax2.loglog(kTlist, upsilon_uplim_yang, linestyle=limit_linestyle_yang.get_linestyle(),color=limit_linestyle_yang.get_color())
ax2.loglog(kTlist, upsilon_lolim_yang, linestyle=limit_linestyle_yang.get_linestyle(),color=limit_linestyle_yang.get_color())
# ax2.loglog(kTlist, upsilon_uplim_ly2, linestyle=limit_linestyle_ly2.get_linestyle(),color=limit_linestyle_ly2.get_color())
# ax2.loglog(kTlist, upsilon_lolim_ly2, linestyle=limit_linestyle_ly2.get_linestyle(),color=limit_linestyle_ly2.get_color())
# ax2.loglog(kTlist, upsilon_uplim_lyboth, linestyle=limit_linestyle_lyboth.get_linestyle(),color=limit_linestyle_lyboth.get_color())
# ax2.loglog(kTlist, upsilon_lolim_lyboth, linestyle=limit_linestyle_lyboth.get_linestyle(),color=limit_linestyle_lyboth.get_color())




# plot all these ratios
#ax3.loglog(kTlist, maz_upsilon/atomdb_upsilon, color=maz_linestyle.get_color())
ax3.loglog(kTlist, yang_upsilon/atomdb_upsilon, color=yang_linestyle.get_color())
#ax3.loglog(kTlist, yangly2_upsilon/atomdb_upsilon, color=yangly2_linestyle.get_color())
#ax3.loglog(kTlist, yanglyboth_upsilon/atomdb_upsilon, color=yanglyboth_linestyle.get_color())
#ax3.loglog(kTlist, dipti_upsilon/atomdb_upsilon, color=dipti_linestyle.get_color())
#ax3.loglog(kTlist, yang_highpt_upsilon/atomdb_upsilon, color=yang_highpt_linestyle.get_color())
#ax3.loglog(kTlist, yang_highpt_bounds_upsilon/atomdb_upsilon, color=yang_highpt_bounds_linestyle.get_color())
#ax3.loglog(kTlist, yang_bounds_upsilon/atomdb_upsilon, color=yang_bounds_linestyle.get_color())
ax3.loglog(kTlist, younger_calc_upsilon/atomdb_upsilon, color=younger_calc_linestyle.get_color())
#ax3.loglog(kTlist, diptifac_upsilon/atomdb_upsilon, color=diptifac_linestyle.get_color())
ax3.loglog(kTlist, yurifac_upsilon/atomdb_upsilon, color=yurifac_linestyle.get_color())
ax3.loglog(kTlist, fursa_frdwa_upsilon/atomdb_upsilon, color=fursa_frdwa_linestyle.get_color())
ax3.loglog(kTlist, fursa_dwa_upsilon/atomdb_upsilon, color=fursa_dwa_linestyle.get_color())
ax3.loglog(kTlist, fursa_ccc_upsilon/atomdb_upsilon, color=fursa_ccc_linestyle.get_color())
ax3.loglog(kTlist, fursa_rccc_upsilon/atomdb_upsilon, color=fursa_rccc_linestyle.get_color())
ax3.loglog(kTlist, fang_upsilon/atomdb_upsilon, color=fang_linestyle.get_color())
#ax3.loglog(kTlist, dipti_upsilon/atomdb_upsilon, color=dipti_linestyle.get_color())

ax3.loglog(kTlist, upsilon_uplim_yang/atomdb_upsilon, color=limit_linestyle_yang.get_color())
ax3.loglog(kTlist, upsilon_lolim_yang/atomdb_upsilon, color=limit_linestyle_yang.get_color())
# ax3.loglog(kTlist, upsilon_uplim_ly2/atomdb_upsilon, color=limit_linestyle_ly2.get_color())
# ax3.loglog(kTlist, upsilon_lolim_ly2/atomdb_upsilon, color=limit_linestyle_ly2.get_color())
# ax3.loglog(kTlist, upsilon_uplim_lyboth/atomdb_upsilon, color=limit_linestyle_lyboth.get_color())
# ax3.loglog(kTlist, upsilon_lolim_lyboth/atomdb_upsilon, color=limit_linestyle_lyboth.get_color())

ax3.set_ylabel('ratio/Dere')
ax2.legend(loc=0)


ax.set_yscale('linear')
ax.set_xscale('linear')

ax.legend(ncol=2, loc=0)


# Ignore all below here

#--
#--p0 = numpy.array([ 24.8, -8.4, 3.8, -20.0])
#--p0*=1e-14
#--enew = numpy.logspace( numpy.log10(I), numpy.log10(max(e_in)*10), 100)
#--
#--bounds = (numpy.array([1e-14, -5e-13, 3.8e-14, -5e-13]),\
#--          numpy.array([5e-13, -1e-14, 3.9e-14, -1e-14]))
#--
#--inxs = evaluate_crosssection_di(enew, p0[0], p0[1], p0[2], p0[3], removemin = True)
#--
#--popt, pcov=  curve_fit(evaluate_crosssection_di,e_in, sig_out, p0=p0)
#--print('no limit')
#--for i in range(len(popt)):
#--  print(p0[i]*1e14, popt[i] * 1e14)
#--
#--nolimcoeff = popt*1
#--nolimxs = evaluate_crosssection_di(enew, popt[0], popt[1], popt[2], popt[3], removemin=True)
#--
#--#popt, pcov=  curve_fit(evaluate_crosssection_di,e_in, sig_out, p0=p0, bounds=bounds)
#--C_Frozen = 3.8e-14
#--bounds_noC = (numpy.array([10, -15, -30]) * 1e-14,\
#--              numpy.array([30, -1, -15]) * 1e-14)
#--
#--popt, pcov=  curve_fit(evaluate_crosssection_di_Cfrozen, e_in, sig_out, p0=[p0[0], p0[1], p0[3]], bounds=bounds_noC)
#--
#--limcoeff = popt*1
#--print('with limit')
#--for i in range(len(popt)):
#--  print(p0[i]*1e14, popt[i] * 1e14)
#--p_freefit = [popt[0], popt[1], C_Frozen *1, popt[2]]
#--#p_freefit = popt
#--#Solutions
#--
#--popt, pcov=  curve_fit(evaluate_crosssection_di_Cfrozen, e_in, sig_out, p0=[p0[0], p0[1], p0[3]])
#--for i in range(len(popt)):
#--  print(p0[i]*1e14, popt[i] * 1e14)
#--p_freefit_nobounds = [popt[0], popt[1], C_Frozen *1, popt[2]]
#--
#--sol={}
#--sol['Fe+25']={}
#--sol['Fe+25']['IONPOT'] = 9278.0
#--sol['Fe+25']['A'] = 13.0e-14
#--sol['Fe+25']['B'] = -4.5e-14
#--sol['Fe+25']['C'] = 1.9e-14
#--sol['Fe+25']['D'] = -10.6e-14
#--
#--sol['Fe+24']={}
#--sol['Fe+24']['IONPOT'] = 8828.0
#--sol['Fe+24']['A'] = 24.8e-14
#--sol['Fe+24']['B'] = -8.4e-14
#--sol['Fe+24']['C'] = 3.8e-14
#--sol['Fe+24']['D'] = -20.0e-14
#--
#--
#--
#--an = [27.0, 60.1, 140.0, -89.9]
#--bn = [-9.62, 33.1, -82.5, 54.6]
#--cn = [3.69, 4.32, -2.527, 0.262]
#--dn = [-21.7, 42.5, -131., 87.4]
#--
#--A, B, C, D = calc_coeffts(an, bn, cn, dn, 26, 2)
#--
#--print("Calculated C coefficient: ", C)
#--zz=input()
#--
#--youngerxs = evaluate_crosssection_di(enew, A, B, C, D, removemin=True)
#--kTlist = numpy.logspace(2,5,100)
#--youngerups = evaluate_ratecoefft(kTlist, A, B, C, D, printme=False)
#--
#--
#--c = numpy.array([I, A, B, C, D])
#--c[1]*=1e14
#--print(c)
#--T = kTlist*11604.5
#--
#--
#--ci = pyatomdb.atomdb._ci_younger(T, c)
#--
#--c_freefit = numpy.append(numpy.array([I]), numpy.array(p_freefit)*1e14)
#--ci_freefit = pyatomdb.atomdb._ci_younger(T, c_freefit)
#--
#--c_freefit_nobounds = numpy.append(numpy.array([I]), numpy.array(p_freefit_nobounds)*1e14)
#--ci_freefit_nobounds = pyatomdb.atomdb._ci_younger(T, c_freefit_nobounds)
#--
#--ax2.plot(kTlist, ci, ':s')
#--ax2.plot(kTlist, ci_freefit, '--^')
#--ax2.plot(kTlist, ci_freefit, '--v')
#--
#--Ebins = numpy.logspace(-1,2+numpy.log10(kTlist[-1]), 100001)
#--
#--tmpyoungerxs = evaluate_crosssection_di(Ebins, A, B, C, D, removemin=True)
#--tmpfreefitxs = evaluate_crosssection_di(Ebins, p_freefit[0], p_freefit[1], p_freefit[2], p_freefit[3], removemin=True)
#--
#--print(max(tmpyoungerxs))
#--ax.plot(Ebins, tmpyoungerxs, '-o')
#--ax.plot(Ebins, tmpfreefitxs, '--x')
#--
#--tmpyoungerxs=(tmpyoungerxs[1:]+tmpyoungerxs[:-1])/2
#--tmpfreefitxs=(tmpfreefitxs[1:]+tmpfreefitxs[:-1])/2
#--youngerupsmymaxwell = numpy.zeros(len(kTlist))
#--freefitupsmymaxwell = numpy.zeros(len(kTlist))
#--
#--sum_fede =  numpy.zeros(len(kTlist))
#--for ikT, kT in enumerate(kTlist):
#--  maxw = my_maxwell(Ebins, kT)
#--  sum_fede[ikT] = sum(maxw)
#--  youngerupsmymaxwell[ikT] = sum(maxw*tmpyoungerxs)
#--  freefitupsmymaxwell[ikT] = sum(maxw*tmpfreefitxs)
#--
#--ax2.plot(kTlist, youngerupsmymaxwell, '-o')
#--ax2.plot(kTlist, freefitupsmymaxwell, '--x')
#--
#--ax3.plot(kTlist, sum_fede)
#--
#--
#--ups_atomdb,zzz = pyatomdb.atomdb.get_ionrec_rate(kTlist/1000, Z=26, z1=25, Te_unit='keV')
#--
#--pyatomdb.util.switch_version('1.3.1')
#--
#--ups_atomdb_131,zzz = pyatomdb.atomdb.get_ionrec_rate(kTlist/1000, Z=26, z1=25, Te_unit='keV')
#--pyatomdb.util.switch_version('3.0.9')
#--
#--ax2.plot(kTlist, ups_atomdb, ':<')
#--ax2.plot(kTlist, ups_atomdb_131, ':>')
#--
#--plt.draw()
#--zzz=input('stop here!')
#--newxs = evaluate_crosssection_di(enew, popt[0], popt[1], C_Frozen, popt[2], removemin=True)
#--
#--
#--ax.plot(e_in, sig_out, 'o')
#--ax.plot(enew, newxs, '-')
#--ax.plot(enew, nolimxs, '-.')
#--ax.plot(enew, inxs, ':')
#--ax.plot(enew, youngerxs, '--')
#--
#--u=numpy.array([1.125, 1.25, 1.50, 2.25, 4.00, 6.00])
#--I_R = 6.4949e2
#--E=u*I_R*13.606
#--QR=numpy.array([0.2866, 0.4878, 0.7429, 1.0245, 1.0393, 0.9249])
#--a0=5.29e-9
#--om=numpy.pi*a0**2/I_R**2 * QR
#--
#--ax.plot(E, om, '^')
#--
#--ups_younger = evaluate_ratecoefft(kTlist, A, B, C, D)
#--
#--print(nolimcoeff)
#--print(A,B,C,D)
#--ups_nolimit = evaluate_ratecoefft(kTlist, nolimcoeff[0], nolimcoeff[1], nolimcoeff[2], nolimcoeff[3])
#--ups_limit = evaluate_ratecoefft(kTlist, limcoeff[0], limcoeff[1], C_Frozen, limcoeff[2])
#--
#--
#--ups_atomdb,zzz = pyatomdb.atomdb.get_ionrec_rate(kTlist/1000, Z=26, z1=25, Te_unit='keV')
#--
#--ups_sol = evaluate_ratecoefft(kTlist, sol['Fe+24']['A'], \
#--                                      sol['Fe+24']['B'], \
#--                                      sol['Fe+24']['C'], \
#--                                      sol['Fe+24']['D'])
#--ax2.plot(kTlist, youngerups,'--')
#--ax2.plot(kTlist, youngerupsmymaxwell,'--s')
#--ax2.plot(kTlist, ups_nolimit,'-.')
#--ax2.plot(kTlist, ups_atomdb,'-x')
#--ax2.plot(kTlist, ups_sol,':')
#--ax2.plot(kTlist, ups_limit,':', color='k')
#--
#--emaxw = numpy.linspace(0,100000, 1000001)
#--
#--upsout = numpy.zeros(len(kTlist))
#--
#--sum_fede = numpy.zeros(len(kTlist))
#--newsig = evaluate_crosssection_di(emaxw, A, B, C, D, removemin=True)
#--newsig[newsig<0] =0.0
#--newsig[numpy.isnan(newsig)] = 0.0
#--print(newsig)
#--newsig= (newsig[:-1]+newsig[1:])/2
#--for ikT, kT in enumerate(kTlist):
#--  maxw = my_maxwell(emaxw, kT)
#--  sum_fede[ikT] = sum(maxw)
#--  upsout[ikT] = sum(maxw*newsig)
#--
#--ax2.plot(kTlist, upsout, '-<')
#--
#--ax3.plot(kTlist, sum_fede, '-<')
#--
#--
#--
#--
plt.draw()
zzz=input('ready')

o = open('log_September.txt', 'w')

o.write("### Cross Sections###\n")
o.write("## Yang data - LyA1\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist[yangly1_sigma>0]
stmp = yangly1_sigma[yangly1_sigma>0]
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## Yang data - LyA2\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist[yangly2_sigma>0]
stmp = yangly2_sigma[yangly2_sigma>0]
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## Yang data - LyA both\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist[yanglyboth_sigma>0]
stmp = yanglyboth_sigma[yanglyboth_sigma>0]
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## DiptiCTMC data\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist[dipti_sigma>0]
stmp = dipti_sigma[dipti_sigma>0]
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## Younger data\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist[younger_sigma>0]
stmp = younger_sigma[younger_sigma>0]
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## DiptiFAC data\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist[diptifac_sigma>0]
stmp = diptifac_sigma[diptifac_sigma>0]
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))


o.write("## YuriFAC data\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist[yurifac_sigma>0]
stmp = yurifac_sigma[yurifac_sigma>0]
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## 1-Sigma - LyA1\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist
stmp = sigma_lolim_ly1
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## 1+Sigma - LyA1\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist
stmp = sigma_uplim_ly1
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## 1-Sigma - LyA2\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist
stmp = sigma_lolim_ly2
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## 1+Sigma - LyA2\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist
stmp = sigma_uplim_ly2
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## 1-Sigma - LyA both\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist
stmp = sigma_lolim_lyboth
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("## 1+Sigma - LyA both\n")
o.write("# Energy (eV)   Sigma (cm2)\n")

etmp = Elist
stmp = sigma_uplim_lyboth
for i in range(len(etmp)):
  o.write("%e %e\n"%(etmp[i], stmp[i]))

o.write("### UPSILONS ###\n")
o.write("## AtomDB Data\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = atomdb_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))


o.write("## Yang Data- LyA1\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = yangly1_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## Yang Data- LyA2\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = yangly2_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## Yang Data- LyA both\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = yanglyboth_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## Dipti CTMC Data\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = dipti_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## Younger Data\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = younger_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## Dipti FAC Data\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = diptifac_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))


o.write("## Yuri FAC Data\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = yurifac_upsilon
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## 1-sigma - Ly1\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = upsilon_lolim_ly1
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## 1+sigma - Ly1\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = upsilon_uplim_ly1
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))



o.write("## 1-sigma - Ly2\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = upsilon_lolim_ly2
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## 1+sigma - Ly2\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = upsilon_uplim_ly2
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))


o.write("## 1-sigma - Ly both\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = upsilon_lolim_lyboth
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.write("## 1+sigma - Ly both\n")
o.write("# Temperature (eV)  Upsilon\n")
ttmp = kTlist
utmp = upsilon_uplim_lyboth
for i in range(len(ttmp)):
  o.write("%e %e\n"%(ttmp[i], utmp[i]))

o.close()
