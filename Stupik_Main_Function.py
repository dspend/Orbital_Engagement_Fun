import numpy as np
from scipy.integrate import odeint
from pyswarm import pso
import matplotlib.pyplot as plt
import math
import stupik_lib

#This is the setup script 
mu=3.986E5;
r=35786+6371;
#Setting it at GEO
ae=0.0343E-3;
nu=math.sqrt(mu/(r**3));

xe=np.matrix([[0],[0],[0],[0],[0],[0]])
xp=np.matrix([[-38.9328],[-100],[10],[0],[0],[0]])
#Initial states for evader (xe) and pursuer (xp)


ae=0.0343*10**-3
ap=0.0686*10**-3
c=3

#Values taken from Stupik for thrusting capabilities

args=(xe,xp,ae,ap,c, nu)
#Passing these through to the optimizer in the args

lb=[-1, -1,-1,0]
ub=[1, 1, 1, 60*60*6]

#Lower and upper bounds on variables

xopt, fopt = pso(stupik_lib.stupik_cost_fun, lb, ub, [], args=args)
print(xopt)
print(fopt)
#It is important to remember that the costates can scale together
#So as long as they all scale the same way (i.e. two times each)
#They are essentially the same 
sol,t_span,control=stupik_lib.stupik_all_out(xopt,*args)

#Plotting Everything
plt.figure(1)
ax = plt.axes(projection='3d')
plt.plot(sol[:, 0],sol[:, 1],sol[:, 2], 'b', label='evader')
plt.plot(sol[:, 12],sol[:,13],sol[:,14], 'r', label='pursuer')
plt.xlabel('Radial (km)')
plt.ylabel('In Track  (km)')
ax.set_zlabel('Cross Track (km)')
ax.legend()
plt.show()

plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(t_span, sol[:, 0],'b', label='evader')
plt.plot(t_span, sol[:, 12],'r', label='pursuer')
plt.xlabel('Time (s)')
plt.ylabel('Radial (km)')
ax.legend()        


plt.subplot(3, 1, 2)
plt.plot(t_span, sol[:, 1],'b', label='evader')
plt.plot(t_span, sol[:, 13],'r', label='pursuer')
plt.xlabel('Time (s)')
plt.ylabel('In Track  (km)')
ax.legend()        



plt.subplot(3, 1, 3)
plt.plot(t_span, sol[:, 2],'b', label='evader')
plt.plot(t_span, sol[:, 14],'r', label='pursuer')
plt.xlabel('Time (s)')
plt.ylabel('Cross Track  (km)')
ax.legend()        
plt.show()


plt.figure(3)
plt.plot(t_span, control[:, 0], label='Radial')
plt.plot(t_span, control[:, 1], label='In Track')
plt.plot(t_span, control[:, 2], label='Cross Track')
plt.xlabel('Time (s)')
plt.legend()
plt.show()      

