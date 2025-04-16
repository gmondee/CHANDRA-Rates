import numpy
import scipy.special
import astropy.io.fits as pyfits
import os
import numpy as np
import glob
import openpyxl

# functions to calculate the ionization rate coefficients due to
# direct collisional ionization (ci) and excitation-autoionization (ea)

# see example at the end

# I haven't made the function for cross sections, which you will need
# to compare the your experiments. 
# 
# Copy them from Urdampilleta, Kaastra & Mehdipour 2017 
# http://www.aanda.org/10.1051/0004-6361/201630170

# example is at the end


KBOLTZ = 8.617385e-8
ME_KEV = 510.99895
APEDDIR = './APED' # THIS IS THE DIRECTORY WHERE YOU HAVE PUT THE IR FILES
DATADIR = '.\csdata'# THIS IS THE DIRECTORY WHERE YOU HAVE PUT THE CROSS SECTION DATA FILES
def calc_ci_urdam(T, param):
  """
  Calculate the collisional electron impact ionization rates using the
  Urdampilletta formulae, for 1 shell

  Parameters
  ----------
  Te : float or array(float)
    Electron temperature (K)
  ionpot : float
    Ionization potential (eV)
  param : array(float)
    The parameters of the equation
    shell no, Ionization potential (keV), 8 parameters

  Returns
  -------
  float or array(float)
    Ionization rate in cm^3 s^-1

  References
  ----------
  Urdampilletta
  """
  import scipy.special
  g7a1 = 3.480230906913262     # sqrt(pi)*(gamma + ln(4))
  g7a2 = 1.772453850905516     # sqrt(pi)
  g7a3 = 1.360544217687075E-02 # 2/147
  g7a4 = 0.4444444444444444    # 4/9
  g7c1 = -4881/8.
  g7c2 =  1689/16.
  p7 = numpy.array([1.000224,-0.113011,1.851039,0.0197311,0.921832,2.651957])
  gamma = 0.577215664901532860606512    #Euler's constant
  a = numpy.array([0.999610841,3.50020361,-0.247885719,0.0100539168,1.39075390e-3,1.84193516,4.64044905])
  r0 = 4.783995473666830e-10   #=2*sqrt(2/pi)*c*1e-18

  ishell = int(param[0])
  eion = param[1] # in keV
  kT = T * KBOLTZ
#  print('eion', eion)
  y = eion/kT
  lam = eion/ME_KEV

  # filter out when y > 200
  yhi = y>200.
  ylo = y<=200.
  en1 = numpy.zeros(len(y), dtype=float)

  en1[ylo] = numpy.exp(y[ylo]) * scipy.special.exp1(y[ylo])
  en1[yhi] = 0.001



  g=numpy.zeros((8, len(kT)))
  #print(y, numpy.exp(y))
  g[0,:] = 1/y
  g[1,:] = en1
  g[2,ylo] = scipy.special.expn(2, y[ylo])*numpy.exp(y[ylo])
  g[2,yhi] = 0.001

  g[3,:] = en1/y
  g[4,:] = (1+en1)/y**2
  g[5,:] = (3+y+2*en1)/y**3

  k = numpy.where(y<0.6)[0]
  if len(k) > 0:
    yy = y[k]
    g[6,k] = numpy.exp(yy) * \
          ((((yy / 486.0 - g7a3) * yy + 0.08) * yy - g7a4) * yy + 4 \
          - (g7a2 * numpy.log(yy) + g7a1) / numpy.sqrt(yy))

  k = numpy.where((y>=0.6) & (y<=20.0))[0]
  if len(k) > 0:
    yy=y[k]
    g[6,k] = (p7[0] + (p7[1] + p7[3] * numpy.log(yy)) /\
              numpy.sqrt(yy) + p7[2] / yy) / (yy + p7[4]) / (yy + p7[5])

  k = numpy.where(y>20.0)[0]
  if len(k) > 0:
    yy=y[k]
    g[6,k] = (((((g7c1 / yy + g7c2) / yy - 22.0) /\
                 yy + 5.75) / yy - 2) / yy + 1)/yy**2


  k = numpy.where(y<0.5)[0]
  if len(k) > 0:
    yy = y[k]
    g[7,k] = (((((-yy / 3000.0 - (1.0 / 384.0)) * yy - (1.0 / 54.0)) * yy +\
             0.125) * yy - 1.) * yy + 0.989056 + \
              (numpy.log(yy) / 2.0 + gamma) * numpy.log(yy)) * numpy.exp(yy)


  k = numpy.where(((y>=0.5) & (y<=20.0)))[0]
  if len(k) > 0:
    yy = y[k]
#    print(yy)
    g[7,k] = ((((a[4] / yy + a[3]) / yy + a[2]) / yy + a[1]) / yy + a[0]) /\
               (yy + a[5]) / (yy + a[6])
  k = numpy.where(y>20.0)[0]
  if len(k) > 0:
    yy = y[k]
    g[7,k] = ((((((13068 / yy - 1764) / yy + 274) / yy - 50) / yy +\
               11) / yy - 3) / yy + 1) / yy**2

  ciout = param[2] * 1e24*(g[0,:] - g[1,:]) + \
          param[3] * 1e24* (g[0,:] - 2 * g[1,:] + g[2,:]) + \
          param[4] * 1e24* (g[3,:] + 1.5 * lam * g[4,:] + 0.25 * lam**2 * g[5,:]) + \
          param[5] * 1e24* g[6,:] +\
          param[6] * 1e24* g[7,:]
  ciout *= r0 * numpy.exp(-y) / eion**2 * y**1.5 * numpy.sqrt(lam)
  ciout[yhi]=0.0
  return(ciout)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def calc_ea_urdam(T, param):
  """
  Calculate the collisional excitation-autoionization rates using the
  Urdampilletta formulae, for 1 shell

  Parameters
  ----------
  Te : float or array(float)
    Electron temperature (K)
  param : array(float)
    The parameters of the equation
    shell no, Ionization potential (keV), 8 parameters

  Returns
  -------
  float or array(float)
    Ionization rate in cm^3 s^-1

  References
  ----------
  Urdampilletta
  """
  import scipy.special
  r0 = 4.783995473666830e-10   #=2*sqrt(2/pi)*c*1e-18

  ishell = int(param[0])
  eion = param[1] # in keV
  kT = T * KBOLTZ
  y = eion/kT
  lam = eion/ME_KEV
  exp1=scipy.special.exp1(y)
  # filter out when y > 200
  yhi = y>200.
  ylo = y<=200.
  #en1 = numpy.zeros(len(y), dtype=float)

  #en1[ylo] = numpy.exp(y[ylo]) * exp1[ylo]
  #en1[yhi] = 0.001

  #eminusy = numpy.exp(-y)

  #m1 = (1/y)* eminusy
  #m2 = exp1
  #m3 = eminusy- y*exp1
  #m4 = (1-y)*eminusy/2 + y**2*exp1/2
  #m5 = exp1/y

  em1 = numpy.zeros(len(y))
  em2 = numpy.zeros(len(y))
  em3 = numpy.zeros(len(y))

  em1[ylo] = scipy.special.expn(1, y[ylo])*numpy.exp(y[ylo])
  em2[ylo] = scipy.special.expn(2, y[ylo])*numpy.exp(y[ylo])
  em3[ylo] = scipy.special.expn(3, y[ylo])*numpy.exp(y[ylo])

  c_ea = param[2]*1e24 + \
         y * (param[3]*1e24*em1 + param[4]*1e24 * em2 + 2*param[5]*1e24 * em3) + param[6]*1e24*em1

  c_ea *=  r0*numpy.exp(-y)/eion**2 * y**1.5 * lam**0.5
  c_ea[yhi]=0.0


  return(c_ea)




def calc_ci_sigma_urdam(E, param):
  """
  Calculate the collisional electron impact ionization rates using the
  Urdampilletta formulae, for 1 shell

  Parameters
  ----------
  E : float or array(float)
    Electron temperature (keV)
  param : array(float)
    The parameters of the equation
    shell no, Ionization potential (keV), 8 parameters

  Returns
  -------
  float or array(float)
    Cross section in cm^2

  References
  ----------
  Urdampilletta
  """
  import scipy.special
  
  I = param[1] # ionization parameter in keV
  A = param[2]
  B = param[3]
  C = param[4]
  D = param[5]
  F = param[6] # this is parameter E in the formula, but I don't want to confuse with "E" for Energy.
  
  tau = I/ME_KEV
  eps = E/ME_KEV
  R = ((tau+2)/(eps+2))* ( (eps+1)/(tau+1))**2 * \
       ( ((tau+eps)*(eps+2) *(tau+1)**2)/ (eps * (eps+2) * (tau+1)**2+tau*(tau+2)))**1.5

  u = E/I
  
  Q = A * (1- (1/u)) + B*(1-(1/u))**2 + C * R * numpy.log(u) + D * numpy.log(u)/numpy.sqrt(u) + F * numpy.log(u)/u
  
  Q = Q/(u*I*I)*10000 # m^2 -> cm^2
  
  # having done all that: at all points, if E < ionization potential (so u < 1), the cross section is 0.
  
  Q[u<1] = 0.0
  return(Q)



def get_params(Z, z1, elsymb):
  import glob
  # Z = element nuclear charge
  # z1 = ionizing ion charge +1 
  # elsymb  = element atomic symbol (e.g. Be, Fe etc)
  
  # the parameters are, for each shell:
  #    the index (so just 1, 2, 3,...),
  #    the ionization potential of the shell (in eV)
  #    the A, B, C, D, E parameters from Urdampilleta.
  
  # Both exciation-autoionization and CI use the same number of and name for 
  # the parameters, but they use different formulae to convert to
  # cross sections/rates
  
  # The bethe limit it parameter C, of CI from shell 1.
  
  
  # get the filename

  searchstring = '%s/%s/%s_%i/%s_%i_IR*fits'%(APEDDIR,
                                             elsymb.lower(),
                                             elsymb.lower(),
                                             z1,
                                             elsymb.lower(),
                                             z1)
                                             
  fname = glob.glob(searchstring)
  print(searchstring, fname)
  irdat = pyfits.open(fname[0])
  
  # get the data:
  
  icidat = numpy.where(irdat[1].data['PAR_TYPE']==52)[0]
  
  cidat = []
  for i in icidat:
    cidat.append(numpy.array(irdat[1].data['IONREC_PAR'][i][:7], dtype=float))

    
  ieadat = numpy.where(irdat[1].data['PAR_TYPE']==64)[0]
  
  eadat = []
  for i in ieadat:
    eadat.append(numpy.array(irdat[1].data['IONREC_PAR'][i][:7], dtype=float))
  

  
  ret={'ci':cidat,
       'ea':eadat}

  return(ret)


def get_ionrec_rate(Z, z1, elsymb, T):
  
  # get the atomic data
  
  dat = get_params(Z, z1, elsymb)
  
  eaout = numpy.zeros(len(T))
  ciout = numpy.zeros(len(T))
  print('dat', dat)
  for ea in dat['ea']:
    eaout += calc_ea_urdam(T, ea )
    
  for ci in dat['ci']:
    print('ci', ci)
    tmp = calc_ci_urdam(T, ci)
    ciout += tmp
    print('tmp', tmp)
  return(ciout, eaout)
    

if __name__ == '__main__':
  import matplotlib.pyplot as plt

  Z = 5
  z1 = 3
  elsymb = 'B'
  Tlist = numpy.logspace(4,10,120) # some temperatures

  searchstringexp = os.path.abspath('%s/%s/%s\%s%i+*xlsx'%(DATADIR,
                                             elsymb.upper(),
                                             "SI",
                                             elsymb.upper(),
                                             z1-1))
  
  fname = glob.glob(searchstringexp)[0]
  print(searchstringexp, fname)

  if fname:
    #load energies, data
    wb = openpyxl.load_workbook(fname)
    ws = wb.active
    ws_contents=list(ws.iter_cols(values_only=True))
    for col in ws_contents:
      if "Energy" in col[0]:
        Elist = col[1:]
      elif col[0] == "cross section":
        csData = col[1:]
      elif col[0] == "cross section uncertainty":
        csUnc = col[1:]
      elif col[0] == "cross section power (cm2)":
        csPower = col[1]

    Elist=np.array(Elist)*10**-3 # in keV
    csData=np.array(csData)*10**csPower
    csUnc =np.array(csUnc)*10**csPower
  else:
    Elist = 2*np.logspace(1,3,num=20)*(10**-3) #in keV
  #Elist = numpy.logspace(-3,1,120)

  ci, ea = get_ionrec_rate(Z, z1, elsymb, Tlist)
  # find the Bethe limit (C)
  params = get_params(Z, z1, elsymb)
  qci = 0.0
  qci_list=[]
  for i in range(len(params['ci'])):
    if int(params['ci'][i][0]) == 1:
       C = params['ci'][i][4]
    print("CIPARAM:",i, params['ci'][i])
    tmp = calc_ci_sigma_urdam(Elist, params['ci'][i])
    print(tmp)
    qci_list.append(tmp)
    qci+=tmp
  print("C is ", C, "m^2 keV^2")
  


  print(ci)
  print(ea)
  fig = plt.figure()
  fig.show()
  ax = fig.add_subplot(111)

#  ax.loglog(Elist, qci, label='ci')
  ax.semilogy(Elist, qci, label='ci')

  for ii,i in enumerate(qci_list):
    ax.semilogy(Elist, i, label='ci:%i'%(ii))
    
#  ax.loglog(Tlist, ea, label='ea')
  ax.set_xlabel("Energy (keV)")
  ax.set_ylabel("Cross section (cm$^2$)")
  ax.errorbar(Elist, csData, csUnc, label='experiment', marker='o')
  
  ax.legend(loc=0)
  plt.draw()
  
  
  
  
  
  
