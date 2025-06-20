import sys
import numpy
import scipy.special
import pylab
import pyatomdb


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
mec2 = 510.99895000 # in keV


def eval_udi_shell(coeff, kT):
  eion = coeff['I_keV']
  y = eion/kT
  lam = eion/mec2
  en1 = numpy.exp(y) * scipy.special.exp1( y)

  g=numpy.zeros((8, len(kT)))
  g[0,:] = 1/y
  g[1,:] = en1
  g[2,:] = scipy.special.expn(2, y)*numpy.exp(y)
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


  shlout =  coeff['A'] * 1e24*(g[0,:] - g[1,:]) + \
      coeff['B'] * 1e24* (g[0,:] - 2 * g[1,:] + g[2,:]) + \
      coeff['C'] * 1e24* (g[3,:] + 1.5 * lam * g[4,:] + 0.25 * lam**2 * g[5,:]) + \
      coeff['D'] * 1e24* g[6,:] +\
      coeff['E'] * 1e24* g[7,:]

  shlout *= r0 * numpy.exp(-y) / eion**2 * y**1.5 * numpy.sqrt(lam)

  return(shlout)

def eval_udi(coeff, kT):
  rate = numpy.zeros(len(kT))
  for i in range(len(coeff)):
#    print(coeff[i])
    rate += eval_udi_shell(coeff[i], kT)

  return(rate)

def eval_uea_shell(coeff, kT, altmethod=False):
  eion = coeff['I_keV']
  y = eion/kT
  print(eion)
  print(kT)
  print(y)
#  zzz=input()
  lam = eion/mec2
  exp1=scipy.special.exp1( y)
  en1 = numpy.exp(y) * exp1
  eminusy = numpy.exp(-y)

#dtype=[('I_keV', '<f8'), ('A', '<f8'), ('B', '<f8'), ('C', '<f8'), ('D', '<f8'), ('E', '<f8')])
  m1 = (1/y)* eminusy
  m2 = exp1
  m3 = eminusy- y*exp1
  m4 = (1-y)*eminusy/2 + y**2*exp1/2
  m5 = exp1/y

  c_ea = m1 * coeff['A']* 1e24 +\
         m2 * coeff['B']* 1e24 +\
         m3 * coeff['C']* 1e24 +\
         m4 * 2 * coeff['D']* 1e24 +\
         m5 * coeff['E']* 1e24

  c_ea *=eminusy*y**1.5*numpy.sqrt(lam)/eion**2

  if altmethod:
    print("ALTMETHOD")

#    m1 =  (1/y)* eminusy
#    m2 = scipy.special.expn(1, y)
#    m3 = scipy.special.expn(2, y)
#    m4 = scipy.special.expn(3, y)
#    m5 = scipy.special.expn(1, y)/y
#    c_ea = m1 * coeff['A']* 1e24 +\
#           m2 * coeff['B']* 1e24 +\
#           m3 * coeff['C']* 1e24 +\
#           m4 * 2 * coeff['D']* 1e24 +\
#           m5 * coeff['E']* 1e24


    em1 = scipy.special.expn(1, y)*numpy.exp(y)
    em2 = scipy.special.expn(2, y)*numpy.exp(y)
    em3 = scipy.special.expn(3, y)*numpy.exp(y)
    emm1 = scipy.special.expn(1, y)
    emm2 = scipy.special.expn(2, y)
    emm3 = scipy.special.expn(3, y)
    #c_ea = coeff['A']*1e24 + y * (coeff['B']*1e24*em1 + coeff['C']*1e24 * em2 + 2*coeff['D']*1e24 * em3) + coeff['E']*1e24*em1

    c_ea = coeff['A']*1e-0 * eminusy/y +\
           coeff['B']*1e-0 * emm1 +\
           coeff['C']*1e-0 * emm2 +\
           2*coeff['D']*1e-0 * emm3 +\
           coeff['E']*1e-0 * emm1/y

#    c_ea *=eminusy*y**1.5*numpy.sqrt(lam)/eion**2
#    print(r0)
#    r0tmp = 2*numpy.sqrt(2/numpy.pi) * kT**-1.5 * mec2**-0.5
#    c_ea *= numpy.exp(-y)
#    print(r0)
    #c_ea *=  numpy.exp(-y)/eion**2 * y**1.5 * lam**0.5
    r0tmp = (2 * numpy.sqrt(2) /\
            ((numpy.pi * kT**3 * mec2 /2.998e8**2)**0.5)) * 1e7

  else:
    print("NORMMETHOD")
    r0tmp=r0*1


  return(r0tmp*c_ea)

def eval_uea(coeff, kT, altmethod=False):
  rate = numpy.zeros(len(kT))
  for i in range(len(coeff)):
#    print(coeff[i])
    rate += eval_uea_shell(coeff[i], kT, altmethod=altmethod)

  return(rate)



def evaluate_crosssection_di_mewe(E, Ain, Bin, Cin, Din, Fin, printme=False, removemin=False):

  A = Ain*1e-14
  B = Bin*1e-14
  C = Cin*1e-14
  D = Din*1e-14
  F = Fin*1e-14

  u=E/I

  mec2 = pyatomdb.const.ME_KEV*1000

  eps = E/mec2
  tau = I/mec2
  R= ((tau+2)/(eps+2)) * ( (eps+1)/(tau+1))**2 * ( ( (tau+eps) * (eps+2) * ((tau+1)**2))/ ( eps*(eps+2)*((tau+1)**2) + tau*(tau+2)))**1.5
  #print("R", R)
  R= 1+1.5*eps +0.25*eps**2
  #print("R2", R2)
  #zzz=input()
  #R=1

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
