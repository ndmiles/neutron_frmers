#!/Users/ndmiles/miniconda3/envs/astroconda27/bin/python
"""
Script for numerically intergratig eq 7 of (Hirsch, 1967)
"""
from scipy import constants
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


#Constants
_m_e = constants.physical_constants['electron mass']
_m_p = constants.physical_constants['proton mass']
_m_n = constants.physical_constants['neutron mass']
_fundamental_charge = constants.physical_constants['atomic unit of charge']
pi = constants.pi


def f(y_,R_,params_): 
    """ Function to be used by scipy.integrate.odeint to solve the 
    system of equations

    Keyword Arguments:
    y_ -- initial conditions
    R_ -- domain of radial coordinates to integrate over 
	  scaled to bein interval [0,1]
    params_ -- variety of parameters present in eq 7.
 
    returns the derivate at each point in the specified domain (R_)
    """
    phi_, E_ = y_
    ranode_ = params_[0]
    Vcathode_ = params_[1]
    rhoi_ = params_[2]
    I_i = params_[3]
    I_e  = params_[4]
    m_e =  params_[5]
    m_i = params_[6]   
    try:
        derivs_ = [E_, -2*E_/R_ + 4 * pi * rhoi_* ranode_**2 * phi_**0.5 \
                  * (1/Vcathode_)*(((np.absolute(phi_))**0.5 - (I_e/I_i)) \
                  * (m_e/m_i) * 1/(1-phi_)**0.5)]
    except Exception as e:
        return e
    else:
        return derivs_



def main():
    #Parameter definition
    phi_ = 0.1
    E_ = 0.5
    R_ = np.arange(0.05,1.05,0.01)
    print(R_)
    #radius of the grounded, outer anode(cm) by definition r_a = r/R in eq 7.
    #Hence it attains a value less than one because we are considering ion motion
    #interior to the outer anode
    ranode_ = 0.8 

    #Cathode voltage (V)
    Vcathode_ = 150000

    #ion charge density
    rho_i = 5

    #ion and electron currents
    I_i = 0.00235
    I_e = 0.00003452

    #electron and ion masses
    m_e = _m_e[0] 
    m_i = _m_p[0]
    y0 = [phi_, E_]
    params_= [ranode_,Vcathode_,rho_i,I_i,I_e,m_e,m_i]
    print(params_)

    psoln = odeint(f,y0,R_,args=(params_,))

    print(psoln)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(R_,psoln[:,0])
    plt.show()

if __name__ == "__main__": 
    main()
