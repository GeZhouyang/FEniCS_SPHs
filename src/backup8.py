get_ipython().magic('matplotlib inline')
get_ipython().magic('run /home/fenics/fenics-matplotlib.py')
from dolfin import *
from mshr import *
from IPython.display import display, clear_output
import logging; logging.getLogger('FFC').setLevel(logging.WARNING)
import time
import numpy as np                                                              
from matplotlib import pyplot as plt   
import sys


print '*** Initialization ***'
print 'Define the geometry and generate the mesh ...'

# Subdomain for the periodic boundary (left/right)

class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)
    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

        
# Subdomain for the free surface at bottom

class freeFace(SubDomain):    
      def inside(self, x, on_boundary):
        return bool(x[0] > 0.25-DOLFIN_EPS and x[0] < 0.75+DOLFIN_EPS and 
                    x[1] < DOLFIN_EPS and on_boundary)
          
        
# Geometry and mesh

XMIN, XMAX, YMIN, YMAX = 0.,1.,0,1.
G = [XMIN, XMAX, YMIN, YMAX]

eps = 1e-7
mres = 128
i_out = 5  # frequency of output

print 'Mesh resolution = ', mres, 'x', mres

if (mres % 4 != 0):
    print "The resolution should be dividable by 4."
    sys.exit()

mesh = UnitSquareMesh(mres, mres)  # better than triangular mesh somehow
h = CellSize(mesh)
hs = (XMAX-XMIN)/mres

mesh_l = IntervalMesh(mres/2, 0.25, 0.75)  # linear mesh (intervals)

    
# Mark the free surface and define its measure

ff = FacetFunction("size_t", mesh)
freeFace().mark(ff, 1)
dsf = Measure("ds")[ff]
    
print 'Define FEM functions ...'
    
# FEM function(space)s

V = VectorFunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
Q = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())      
W = V * Q
w = Function(W)
(u, p) = (as_vector((w[0], w[1])), w[2])   # velocity and pressure
(v, q) = TestFunctions(W)

n_V = V.dim(); d_V = mesh.geometry().dim()
#print n_V,d_V
stress_m = Function(V)  # modified stress

# mesh coordinates
V_dof_coord = V.dofmap().tabulate_all_coordinates(mesh)
V_dof_coord.resize((n_V, d_V))
x_m = V_dof_coord[:, 0]
y_m = V_dof_coord[:, 1]

# Buffer in space V
bufv = np.zeros(n_V)  
    
# index list (in V) of the free surface
ind_list = []  
for i in range(len(bufv)): 
    #print 'look',i,bufv[i],x_m[i],y_m[i]
    if (x_m[i] > 0.25-eps and x_m[i] < 0.75+eps):
        ind_list.append(i)


Vx = FunctionSpace(mesh, "CG", 1)  
ux = Function(Vx)  # x-component velocity

Z_l = FunctionSpace(mesh_l, "CG", 1)
n_l = Z_l.dim(); d_l = mesh_l.geometry().dim()
#print 'n and d', n_l, d_l
c_l = Function(Z_l)  # surface concentration
c_l0 = Function(Z_l)
z_l = TestFunction(Z_l)

Z_lv = VectorFunctionSpace(mesh_l, "CG", 1)
u_l = Function(Z_lv)  # surface velocity
z_lv = TestFunction(Z_lv)

# surface mesh coordinates
Z_l_dof_coord = Z_l.dofmap().tabulate_all_coordinates(mesh_l)
Z_l_dof_coord.resize((n_l, d_l))
x_l = Z_l_dof_coord[:, 0]
    
# Obtain the vertex indices corresponding to the free surface
vs = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(ff, 1)), [])))
v2d = vertex_to_dof_map(Vx)
d = v2d[vs]


# Plot the velocity contour

def plotta(stepcounter,u):
    if stepcounter % 5 == 0:
        uEuclidnorm = project(sqrt(inner(u, u)), Q)
        ax.cla(); fig = plt.gcf();
        #plt.subplot(2, 1, 1);
        mplot_function(uEuclidnorm)   
        plt.title("Velocity magnitude") # Plot norm of the velocity
        if stepcounter == 0.:
            plt.colorbar();
            plt.axis(G)
        plt.axes().set_aspect('equal')
        
        #plt.subplot(2, 1, 2);
        #if stepcounter == 0.:
        #    plt.triplot(mesh2triang(mesh), color='#000000');
        #    plt.title("Mesh") # Plot mesh
        
        plt.show()

        
# Extract the surface velocity and load it into the 1D mesh

def extract_surf_vel(i,ux):
        
    # Extract the tangential velocity along the free surface
    vel = ux.vector().array()[d]  
    #for i in range(len(vel)): print i,vel[i]  
        
    # Load surface velocity
    buffer_l = np.zeros(n_l)  
    buffer_l = vel
    u_l.vector()[:] = buffer_l 
    
    # Plot to verify
    if ((i+1) % i_out == 50):
        plt.plot(x_l, u_l.vector()[:])
        plt.title("Surface velocity, i = "+str(i+1))
        plt.xlabel("x")
        plt.xlim(0.25,0.75)    
        plt.show()
    
    return u_l


# Compute the tangential shear stress

def comp_stress(ik,c_l):
            
    # Compute the tangential stress using the Langmuir isotherm
    for i in range(len(ind_list)): 
        ii = ind_list[i]
        xx = x_m[ii]  # x-coordinate
        #print 'check',i,xx
        
        j = [itemp for itemp in range(len(x_l)) if x_l[itemp] == xx]
        jp = j[0] +1
        jm = jp-2
        
        if (xx > 0.25 and xx < 0.75): # non-zero gradient if inside
            gradc = (c_l[jp]-c_l[jm])/(x_l[jp]-x_l[jm])  # gradient of c
            bufv[ii] = beta/(1.-c_l[j])*gradc
                
        #print 'Jo',i,xx,bufv[ii]        
        
    # Update the tangential stress
    stress_m.vector()[:] = bufv
    
    return stress_m


# Boundary markers

top = Expression("x[1] > YMAX - eps ? 1. : 0.", YMAX = YMAX, eps=eps) 
bottom = Expression("x[1] < YMIN + eps ? 1. : 0.", YMIN = YMIN, eps=eps)
bottom_wall = Expression("(x[0] < 0.25 + eps || x[0] > 0.75 - eps) && x[1] < YMIN + eps ? 1. : 0.", 
                         YMIN = YMIN, eps=eps)
bottom_free = Expression("x[0] > 0.25 && x[0] < 0.75 && x[1] < YMIN + eps ? 1. : 0.", 
                         YMIN = YMIN, eps=eps)


# Prescribed expressions

unit_tang = Constant(("1.","0."))
unit_norm = Constant(("0.","1."))
fun0 = Constant(("0.","0."))
fun1 = Expression(("-10.*(x[0]-XMAX/10.)", "0."), XMAX=XMAX)
fun2 = Expression(("-exp(x[0]-0.25)", "0."), XMAX=XMAX)


################## Physical parameters ##################

Pe = 5e-1  # bigger Pe, weaker diffusion
print '* Peclet number = ', Pe

beta = 1e0  # bigger beta, stronger effect of surfactant
print '* Interfacial elasticity = ', beta

c00 = 0.4 # initial surfactant concentration 
          # relative to saturation value

#########################################################


# initialize the concentration (in the whole domain)

c_l0.interpolate(Constant(c00))
c_l0_avr = assemble(c_l0*dx)/0.5
print '* Initial (uniform) surfactant concentration = ', c_l0_avr

print '\nIterations start!'

# Numerical parameters

i = 0     # iteration counter
i_max = 50  # max number of iterations
itol = []  # stopping list
i_sub_max = 10  # max number of sub-iterations
dt = 0.1  # pseudo time step
sub_tol = 0.5*hs  # sub-iteration tolerance

gamma = 1e6*1./h # penalty parameter
eff_slip = []  # effective slip length

pl, ax = plt.subplots();
timer0 = time.clock()

# Looping

while i < i_max:
    
    print '\n** Iteration = ', i+1
    
    # Weak residual of the incompressible Stokes
    r = (-inner(grad(u),grad(v)) + p*div(v) + div(u)*q)*dx
    r += 0.2*h*h*inner(grad(q), grad(p))*dx  # stabilization (if linear elements)
    
    # Boundary conditions
    r += top*(inner(unit_tang,v) + gamma*p*q)*ds               # shear-driven on top (and zero pressure)
    r += bottom_free*(-inner(stress_m,v) + gamma*u[1]*v[1])*ds # prescribed stress + zero v along the free surface
    r += bottom_wall*gamma*inner(u,v)*ds                       # no slip along the wall
    
    # Solution of u and p
    solve(r == 0, w)
    
    ux = project(inner(u, unit_tang), Vx)  # x-dir velocity
    u_l = extract_surf_vel(i,ux)           # surface velocity
    
    # Some output    
    u_avr = assemble(top*ux*ds)/(XMAX-XMIN)
    print 'Average velocity along the top boundary = ', u_avr    
    u_slip = assemble(ux*dsf(1))/0.5
    print 'Average velocity along the free surface = ', u_slip
    print 'Begin sub-iterations'
        
    # Pseudo time-stepping    
    for i_sub in range(i_sub_max): 
        
        c_lm = 0.5*(c_l0 + c_l)  # 2nd order Crank-Nicolson
    
        # Weak residual of the advection-diffusion eq.
        r_c = inner(c_l - c_l0,z_l)/dt*dx + (1./Pe)*inner(grad(c_lm),grad(z_l))*dx + div(c_lm*u_l)*z_l*dx
        
        # Solution of c
        solve(r_c == 0, c_l)
    
        # Relative error norm
        dc_l = c_l-c_l0
        cnorm = assemble(abs(dc_l)*dx)/0.5/c_l0_avr
        print 'Step = ', i_sub+1, ' L1 norm = ', cnorm
            
        i_sub += 1
        c_l0 = project(c_l, Z_l) 
        
        # Mass correction
        if (i_sub % 3 == 0):
            c_l_avr = assemble(c_l*dx)/0.5
            c_l0 += -(c_l_avr-c_l0_avr)
        
        if (cnorm < sub_tol): break
            
    # Mass correction
    c_l_avr = assemble(c_l*dx)/0.5
    print 'Surfactant relative mass loss = ', (c_l_avr-c_l0_avr)/c_l0_avr
    c_l0 += -(c_l_avr-c_l0_avr)
        
    # Plot to verify
    if ((i+1) % i_out == 0):
        plt.plot(x_l, u_l.vector()[:])
        plt.title("Surface velocity, i = "+str(i+1))
        plt.xlabel("x")
        plt.xlim(0.25,0.75)    
        plt.show()
        
        plt.plot(x_l, c_l.vector())
        plt.title("Surfactant distribution, i = "+str(i+1))
        plt.xlabel("x")
        plt.xlim(0.25,0.75)
        plt.show()
    
    # Compute the tangential shear stress    
    stress_m = comp_stress(i,c_l.vector())
    
    # Effective slip length
    u_avr = assemble(top*ux*ds)/(XMAX-XMIN)
    eff_slip.append(u_avr-1.)
    
    # Check convergence
    if (i > 0):
        rel_change = abs((eff_slip[i] - eff_slip[i-1])/eff_slip[i-1])
        if (rel_change < 0.03): 
            itol.append(1)
        else:
            itol.append(0)
        
    # Next iteration
    xkcd = len(itol)
    if (xkcd > 2 and sum(itol[i] for i in range(xkcd-3,xkcd)) == 3):
        break  # stop if the last three entries are within 3%
    else:        
        i += 1
    
    
# Final output & plotting
if ((i+1) % i_out != 0):
    plt.plot(x_l, u_l.vector()[:])
    plt.title("Surface velocity, i = "+str(i+1))
    plt.xlabel("x")
    plt.xlim(0.25,0.75)    
    plt.show()
        
    plt.plot(x_l, c_l.vector())
    plt.title("Surfactant distribution, i = "+str(i+1))
    plt.xlabel("x")
    plt.xlim(0.25,0.75)
    plt.show()
        
plt.plot([k+1 for k in range(len(eff_slip))],eff_slip,'--s')
plt.title("Effective slip length")
plt.xlabel("Iterations")
plt.ylim(0,0.1)
plt.show()
print '\nFinal effective slip lengh = ', eff_slip[-1]

# Velocity contour
plotta(0,u)

plt.close();
print "\nelapsed CPU time: ", (time.clock() - timer0)
print '*** Completed ***'
