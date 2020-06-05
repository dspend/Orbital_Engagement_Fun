import numpy as np
from scipy.integrate import odeint
from pyswarm import pso
import matplotlib.pyplot as plt
import math
import stupik_lib

#This is the setup script 
mu=3.986E5; #Earth Gravity Parameter (km)
r=35786+6371;   #Semi major axis of the virtual chief (km)
#Setting it at GEO for this example
nu=math.sqrt(mu/(r**3)); #Mean motion of virtual chief

xe=np.matrix([[0],[0],[0],[0],[0],[0]]) #Starting position of evader in RIC km
xp=np.matrix([[-38.9328],[-100],[10],[0],[0],[0]])  #Starting position of Pursuer in RIC km
#Initial states for evader (xe) and pursuer (xp)

ae=0.0343*10**-3 #Instanteous thrust (km/s) of evader
ap=0.0686*10**-3 #Instanteous thrust (km/s) of pursuer
c=3 #Used in Stupiks control law

#Values taken from Stupik for thrusting capabilities

sout=stupik_lib.stupik_json_out(xe, ae, xp, ap, c, nu)
