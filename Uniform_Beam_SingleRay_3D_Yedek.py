# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:29:31 2021

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


#SIMULATION SETUP

m0 = 4*math.pi*1e-7 #magnetic permeability of vacuum [H/m]
e0 = 8.8541878176e-12 #electric permittivity of vacuum [F/m]
c0 = math.sqrt(1/(m0*e0)) #speed of light in vacuum [m/s]

ni=1.33
nt=1.45 #silica particle refractive index

Pi=0.1 #W

#Simulation boundaries
x=200*1e-6#m X length
y=200*1e-6#m Y length

#Ray number
Rx= 7200#Ray number along x
Ry= 7200#Ray number along y
Rsum=Rx*Ry#Total number of ray

#Determine Δx and Δy between rays
delx= x/Rx  #Δx
dely= y/Ry  #Δy

#Indicate Particle Position
posx=+50*1e-6#m
posy=+80*1e-6#m
#Origin is O[0,0] particle Pos: P[50µm,80µm]

#Convert Particle Position to Ray coordinate
Rayx= int(posx//delx) #Ray x_coordinate
Rayy= int(posy//dely) #Ray x_coordinate
#Ray Coordinate of center of Particle: Rc[1800,2880]

#Indicate particle radius
r_p=5*1e-6#m

#Find boundary ray coordinates of particle
delRb_x= r_p/delx #boundary ray's distance from particle center in terms of ray coordinates
delRb_y= r_p/dely
#180

Rb_x=Rayx+delRb_x 
mRb_y=Rayy+delRb_y
mRb_x=Rayx-delRb_x
mRb_y=Rayy-delRb_y

"""
#Shape a rectangle around particle
Rec1=[mRb_x,Rb_y]
Rec2=[Rb_x,Rb_y]
Rec3=[Rb_x,mRb_y]
Rec4=[mRb_x,mRb_y]
"""
#Ray Force Function
def RayForce(i):
    deg_i=i    
    rad_i=(math.pi/180)*deg_i
    rad_t= math.asin((ni*math.sin(rad_i))/nt)    
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
    
    Fs= Qs*Pi*ni/c0 #along z- direction
    
    Fg= -(Qg*Pi*ni/c0)
    
    Fnet= math.sqrt((Fs**2)+(Fg**2))
    
    #Find Gradient Force x and y components
    #Gradient Vector x comp: [Rax-Rcx], y comp: [Ray-Rcy]
    #Find unit vectors
    if ra==0:
        Fgx=0
        Fgy=0
    else:
        Rgx_unit= (Rax-Rcx)/ra
        Rgy_unit= (Ray-Rcy)/ra
    
        if Rgx_unit == 0:
            Fgx=0
            Fgy=Fg
        else:
            Fgx=Fg/(Rgx_unit+(Rgy_unit*(Rgy_unit/Rgx_unit)))
            Fgy= (Rgy_unit/Rgx_unit)*Fgx
   
#    Fsf.append(Fs)
#    Fgf.append(abs(Fg))
#    Fnetf.append(Fnet)
    return (Fs,Fg,Fgx,Fgy)


#Find the Force of single Ray when particle is at posx,posy
#Ray coordinate Ra[1700,3000] in terms of ray coordinates
Rax=1800
Ray=2880

Rcx=Rayx#Particle center ray coordinate
Rcy=Rayy#Particle center ray coordinate
#Rc[1800,2880]
#First check whether Ray is on particle or not (For spherical particle)
ra=math.sqrt(((Rax-Rcx)**2)+((Ray-Rcy)**2)) 

if ra< delRb_x:
    print("Ray is on the particle")
    #Take proportion according to 90deg.-> convert ray to theta 
    theta=(ra*90)/delRb_x
    print("Ray Force Gradient:",RayForce(theta)[1])
    print("Ray Force Gradient_x:",RayForce(theta)[2])
    print("Ray Force Gradient_y:",RayForce(theta)[3])
    print("Ray Force Scattering:",RayForce(theta)[0])
    
    

else:
    print("Ray is not on the particle")

