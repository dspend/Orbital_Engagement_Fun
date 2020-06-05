import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import json
from pyswarm import pso

#These are all just functions I can call in a seperate script

def stupik_initial(costates_0,nu, tf):
    #Program to find the three remaning initial costates using the first of
    #Design variables. This was solved by Stupik in her thesis, and show that
    #Both the pursuer's and evader's costates, and therefore controls in a
    #linear time invariant system are the same.
    #Python isn't great a straight matrix math its a little messy

    #This is all covered in section 3.2.3 of her thesis
    eta=tf-0

    phi=np.matrix([[4-3*math.cos(nu*eta), 6*nu*eta-6*math.sin(nu*eta), 0, -3*nu*math.sin(nu*eta), -6*nu*(1-math.cos(nu*eta)), 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, math.cos(nu*eta), 0, 0, nu*math.sin(nu*eta)],
                       [-math.sin(nu*eta)/nu, -2*(1-math.cos(nu*eta))/nu, 0, math.cos(nu*eta), 2*math.sin(nu*eta), 0],
                       [2*(1-math.cos(nu*eta))/nu, (3*nu*eta-4*math.sin(eta*nu))/nu, 0, -2*math.sin(nu*eta), 4*math.cos(nu*eta)-3, 0],
                       [0, 0, -math.sin(nu*eta)/nu, 0, 0, math.cos(nu*eta)]])

    P=np.zeros([12,12])

    P[0:6,0:6]=phi
    P[6:12,6:12]=np.matrix(phi)

    Gamma=np.zeros([9,9])
    Gamma[0:6,0:3]=P[0:6,3:6]
    Gamma[0:3,3:9]=P[6:9,6:12]
    Gamma[6:9,3:9]=P[9:12,6:12]

    C=np.zeros([9,9])
    C[0:6,0:4]=np.matrix(P[0:6,0:4])

    costates=np.zeros([9,1])
    costates[0]=costates_0[0]
    costates[1]=costates_0[1]
    costates[2]=costates_0[2]
    Costates_Rest=inv(Gamma)
    Costates_Rest=np.dot(Costates_Rest,C)
    Costates_Rest=np.dot(Costates_Rest,costates)

    costates_0=Costates_Rest[3:9]

    
    return costates_0

def stupik_ode(vec,t,xe,xp,ae,ap,c, nu ):
    #This is the ordinary differential equation setup
    #It uses the classical HCW equations of motion
    #With the same control law as Stupik (accounts for mass loss)
    #Due to Matrix Multiplation problems there is a lot of unravelling
    #I unravelled until it ran
    #Python is weird in its indexing because I think it is less than
    #So 0:6 is the first 6 numbers, which is weird because it starts
    #At zero but oh well, it works
    #It goes Evader states (x,y,z,xdot,ydot,zdot), costates, pursuer states
    #Again, pursuer and evader share costates
    
    n=nu
    ex = vec[0]
    ey = vec[1]
    ez = vec[2] 
    edxdt = vec[3]
    edydt = vec[4]
    edzdt = vec[5]

    xe=vec[0:6]
    xe=xe.T
    costatex = vec[6]
    costatey = vec[7]
    costatez = vec[8]
    costatedxt = vec[9]
    costatedydt = vec[10]
    costatedzdt = vec[11]
    costates=vec[6:12]
    px = vec[12]
    py = vec[13]
    pz = vec[14] 
    pdxdt = vec[15]
    pdydt = vec[16]
    pdzdt = vec[17]


    xp=vec[12:18]
    A = np.matrix([[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [3*n*n, 0, 0, 0, 2*n, 0],
         [0, 0, 0, -2*n, 0, 0],
         [0, 0, -n*n, 0, 0, 0]])

    # HCW A matrix
    A_transpose=-A.transpose()

    norm_costates=math.sqrt(costatedxt*costatedxt+costatedydt*costatedydt+costatedzdt*costatedzdt)
    Sin_B=costatedzdt/norm_costates
    Cos_B=math.cos(math.asin(Sin_B))

    Cos_A=costatedydt/(Cos_B*norm_costates)

    Sin_A=costatedxt/(Cos_B*norm_costates)
    #Equations 3.22 from Stupik
    
    B=np.matrix([[0],[0],[0],[Sin_A*Cos_B],[Cos_A*Cos_B],[Sin_B]]); 
    dotXe=A.dot(xe)
    dotXe=dotXe.T+B*ae/(1-t*ae/c)
    dotcostate=np.dot(-A_transpose,costates)
    dotXp=A.dot(xp)
    dotXp=dotXp.T+B*ap/(1-t*ap/c)
    dotVec=np.zeros([1,18])
    dotVec=np.vstack((dotXe,dotcostate.T))
    dotVec=np.vstack((dotVec,dotXp))
    dotVec=np.ravel(dotVec)
    #States costates in a array so that odeint is happy
    return dotVec


def stupik_cost_fun(design_variables,*args):
    #This is the main function to call by the optimizer

    xe,xp,ae,ap,c, nu = args
    args=(xe,xp,ae,ap,c, nu)
    #unpacking the arguments
    
    costates_tf=design_variables[0:3]
    tf=design_variables[3]
    #This is the design variables, three initial costates and final time
   
    costates_0=stupik_initial(costates_tf,nu, tf)
    #Calculating the remaining 3 intial costates
    vector=np.zeros([18,1])
    vector[0:6,0]=np.matrix.flatten(xe)
    vector[6:12,0]=np.matrix.flatten(costates_0)
    vector[12:18,0]=np.matrix.flatten(xp)
    vector=np.ravel(vector)

    #Setting up the initial states for odeint
    t_span=np.linspace(0,np.ravel(tf),2)
    t_span=np.ravel(t_span)
    sol = odeint(stupik_ode, vector, t_span, args)
    #Propogating it out
    cost=sol[-1,0:3]-sol[-1,12:15]
    #Finding the difference in the last (-1) index of the evader and purser
    #position
    cost=cost[0]**2+cost[1]**2+cost[2]**2
    #Taking the norm and setting it as the cost
    return cost 

def stupik_all_out(design_variables,*args):
    #This function takes the design variables and produces the solution
    #For plotting purposes
    #It follows the same path as the cost function but provides the sol
    #instead of the norm
    #Solution has 
    xe,xp,ae,ap,c, nu = args
    args=(xe,xp,ae,ap,c, nu)

    costates_tf=design_variables[0:3]
    tf=design_variables[3]
   
    costates_0=stupik_initial(costates_tf,nu, tf)

    vector=np.zeros([18,1])
   
    vector[0:6,0]=np.matrix.flatten(xe)
    vector[6:12,0]=np.matrix.flatten(costates_0)
    vector[12:18,0]=np.matrix.flatten(xp)
    vector=np.ravel(vector)
    t_span=np.linspace([1],np.ravel(tf),1000)
    t_span=np.ravel(t_span)
    sol = odeint(stupik_ode, vector, t_span, args)

    count=0
    for x in t_span:
        costatedxt=sol[count,10]
        costatedydt=sol[count,11]
        costatedzdt=sol[count,12]
        norm_costates=math.sqrt(costatedxt*costatedxt+costatedydt*costatedydt+costatedzdt*costatedzdt)
        Sin_B=costatedzdt/norm_costates
        Cos_B=math.cos(math.asin(Sin_B))
        Cos_A=costatedydt/(Cos_B*norm_costates)
        Sin_A=costatedxt/(Cos_B*norm_costates)
        #Equations 3.22 from Stupik
        #Saving each control vector away for plotting since ODE doesn't
        if count ==0:
            control=[Sin_A*Cos_B,Cos_A*Cos_B,Sin_B]
        else:
            control=np.vstack([control,[Sin_A*Cos_B,Cos_A*Cos_B,Sin_B]])
        count=count+1
        
    return sol,t_span,control

def stupik_json_out(xe, ae, xp, ap, c, nu):

    args=(xe,xp,ae,ap,c, nu)



    args=(xe,xp,ae,ap,c, nu)
    #Passing these through to the optimizer in the args

    lb=[-1, -1,-1,0]
    ub=[1, 1, 1, 60*60*6]

    #Lower and upper bounds on variables

    xopt, fopt = pso(stupik_cost_fun, lb, ub, [], args=args)
    #It is important to remember that the costates can scale together
    #So as long as they all scale the same way (i.e. two times each)
    #They are essentially the same 
    sol,t_span,control=stupik_all_out(xopt,*args)

    stout={'sol':[sol],'t_span':[t_span],'control':[control],'xopt':[xopt],'fopt':[fopt]}
    return stout
