# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:40:26 2021

@author: Berk
"""


"""
Optical Force (Ray Optics)
"""

"""
F=(n*P/c)*Q

F: Optical Force
n: Refractive index
P: Beam Power
c: Speed of light in vacuum
Q: Trapping efficiency
"""

import math
import numpy as np
import pylab as plt
import matplotlib.pyplot as plt1




m0 = 4*math.pi*1e-7 #magnetic permeability of vacuum [H/m]
e0 = 8.8541878176e-12 #electric permittivity of vacuum [F/m]
c0 = math.sqrt(1/(m0*e0)) #speed of light in vacuum [m/s]

ni=1.33
nt=1.45 #silica particle refractive index

Pi=0.1 #W



#Final lists
deg_if=[]
Qsf=[]
Qgf=[]
Fsf=[]
Fgf=[]
Fnetf=[]

for i in np.arange(0,90,1):    

    deg_i=i
    
    deg_if.append(i)
    rad_i=(math.pi/180)*deg_i
    rad_t= math.asin((ni*math.sin(rad_i))/nt)
    
    #print(rad_t)
    
    theta_i=rad_i
    theta_t=rad_t
    
    
    Rs= (abs(((ni*math.cos(theta_i))-(nt*math.cos(theta_t)))/((ni*math.cos(theta_i))+(nt*math.cos(theta_t))))**2)
    
    Rp= (abs(((ni*math.cos(theta_t))-(nt*math.cos(theta_i)))/((ni*math.cos(theta_t))+(nt*math.cos(theta_i))))**2)
    
    Ts=(4*ni*nt*math.cos(theta_i)*math.cos(theta_t))/(abs(((ni*math.cos(theta_i))+(nt*math.cos(theta_t))))**2)
    
    Tp=(4*ni*nt*math.cos(theta_i)*math.cos(theta_t))/(abs(((ni*math.cos(theta_t))+(nt*math.cos(theta_i))))**2)
    
    #Rs+Ts=1
    #Rp+Tp=1
    
    R=(Rs+Rp)/2
    T=(Ts+Tp)/2
    
    
    Qs=1+ (R*math.cos(2*theta_i))-(T**2)*((math.cos((2*theta_i)-(2*theta_t)))+(R*math.cos(2*theta_i)))/(1+(R**2)+2*R*math.cos(2*theta_t))
    
    Qg=(R*math.sin(2*theta_i))-(T**2)*((math.sin((2*theta_i)-(2*theta_t)))+(R*math.sin(2*theta_i)))/(1+(R**2)+2*R*math.cos(2*theta_t))
    
    Qsf.append(Qs)
    Qgf.append(abs(Qg))
    
    Fs= Qs*Pi*ni/c0
    
    Fg= Qg*Pi*ni/c0
    
    Fnet= math.sqrt((Fs**2)+(Fg**2))
    
    Fsf.append(Fs)
    Fgf.append(abs(Fg))
    Fnetf.append(Fnet)


#ang=54 #angle you want
#
#print("F Scattering: ",Fsf[ang],"N")
#print("F Gradient: ",Fgf[ang],"N")
#print("F Net: ",Fnetf[ang],"N")

"""
#Qs,Qg plot
x1 = deg_if
y1= Qsf

plt.plot(x1, y1)

x2 = deg_if
y2= Qgf

plt.plot(x2, y2)
"""

"""
ang=80 #angle you want

print("F Scattering: ",Fsf[ang],"N")
print("F Gradient: ",Fgf[ang],"N")
print("F Net: ",Fnetf[ang],"N")
"""

##Fs,Fg,Fnet plot
#x1 = deg_if
#y1= Fsf
#
#plt.plot(x1, y1)
#
#x2 = deg_if
#y2= Fgf
#
#plt.plot(x2, y2)
#
#x3 = deg_if
#y3= Fnetf
#
#plt.plot(x3, y3)
"""
V = np.array([[Fsf[ang]*1,Fsf[ang]*0], [Fgf[ang]*0,Fgf[ang]*1], [Fsf[ang]*1,Fgf[ang]*1]])
origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point

plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=1e-9)
v12 = V[0] + V[1] # adding up the 1st (red) and 2nd (blue) vectors
#plt.quiver(*origin, v12[0], v12[1])
plt.show()
"""

#Final lists
mdeg_if=[]
mQsf=[]
mQgf=[]
mFsf=[]
mFgf=[]
mFnetf=[]

for i in np.arange(0,-90,-1):    
    deg_i=i
    mdeg_if.append(i)
    rad_i=(math.pi/180)*deg_i
    rad_t= math.asin((ni*math.sin(rad_i))/nt)
    
    #print(rad_t)
    
    theta_i=rad_i
    theta_t=rad_t
    
    
    Rs= (abs(((ni*math.cos(theta_i))-(nt*math.cos(theta_t)))/((ni*math.cos(theta_i))+(nt*math.cos(theta_t))))**2)
    
    Rp= (abs(((ni*math.cos(theta_t))-(nt*math.cos(theta_i)))/((ni*math.cos(theta_t))+(nt*math.cos(theta_i))))**2)
    
    Ts=(4*ni*nt*math.cos(theta_i)*math.cos(theta_t))/(abs(((ni*math.cos(theta_i))+(nt*math.cos(theta_t))))**2)
    
    Tp=(4*ni*nt*math.cos(theta_i)*math.cos(theta_t))/(abs(((ni*math.cos(theta_t))+(nt*math.cos(theta_i))))**2)
    
    #Rs+Ts=1
    #Rp+Tp=1
    
    R=(Rs+Rp)/2
    T=(Ts+Tp)/2
    
    
    Qs=1+ (R*math.cos(2*theta_i))-(T**2)*((math.cos((2*theta_i)-(2*theta_t)))+(R*math.cos(2*theta_i)))/(1+(R**2)+2*R*math.cos(2*theta_t))
    
    Qg=(R*math.sin(2*theta_i))-(T**2)*((math.sin((2*theta_i)-(2*theta_t)))+(R*math.sin(2*theta_i)))/(1+(R**2)+2*R*math.cos(2*theta_t))
    
    mQsf.append(Qs)
    mQgf.append(-Qg)
    
    Fs= Qs*Pi*ni/c0
    
    Fg= Qg*Pi*ni/c0
    
    Fnet= math.sqrt((Fs**2)+((-Fg)**2))
    
    mFsf.append(Fs)
    mFgf.append(-Fg)
    mFnetf.append(Fnet)

ang0=80 #angle you want
ang=abs(ang0)

if ang0<0 :


    
    print("F Scattering: ",mFsf[ang],"N")
    print("F Gradient: ",mFgf[ang],"N")
    print("F Net: ",mFnetf[ang],"N")
else:


    
    print("F Scattering: ",Fsf[ang],"N")
    print("F Gradient: ",Fgf[ang],"N")
    print("F Net: ",Fnetf[ang],"N")
    

Fsbeam= [] #Scattering Force of uniform beam along x or y direction 
for i in range(0,90):
    Fsbeam.append(Fsf[i]+mFsf[i])

Fgbeam= [] #Gradient Force of uniform beam along x or y direction
for i in range(0,90):
    Fgbeam.append(Fgf[i]+mFgf[i])

Fnetbeam= [] #Net Force of uniform beam along x or y direction
for i in range(0,90):
    Fnetbeam.append(Fnetf[i]+mFnetf[i])


#Print net force of Beam

print("F_Beam Scattering: ",Fsbeam[ang],"N")
print("F_Beam Gradient: ",Fgbeam[ang],"N")
print("F_Beam Net: ",Fnetbeam[ang],"N")

V = np.array([[Fsbeam[ang]*0,Fsbeam[ang]*(-1)], [Fgbeam[ang]*1,Fgbeam[ang]*0], [Fgbeam[ang]*1,Fsbeam[ang]*(-1)]])
origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point

plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=1e-9)
v12 = V[0] + V[1] # adding up the 1st (red) and 2nd (blue) vectors
#plt.quiver(*origin, v12[0], v12[1])
plt.show()



w0=90*(1e-6) #beam waist
length=200*(1e-6) #sample length
r=5*(1e-6)# particle radius
ray= (length/r)*90


#length/ray  distance of one ray


#I= math.exp((r**2)/(2*(w0**2)))


beta=[]
for i in range(0,1800):
    ray_dist= (length/ray) * i #distance of i'th ray from origin
    
    I= math.exp(2*(-ray_dist**2)/((w0**2)))
    beta.append(I)