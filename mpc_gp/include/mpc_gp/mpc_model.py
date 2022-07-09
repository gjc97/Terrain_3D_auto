import numpy as np
from regex import P
import forcespro
import forcespro.nlp
import casadi
import sys
import os
import rospkg
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('mpc_gp')



class GPMPCModel:
    def __init__(self,model_build = False, N = 20, dt = 0.1, Q = None, R = None, solver_name = "MPCGPSOLVER", point_reference=False):
        try:
            # solver_dir = pkg_dir+"/FORCESNLPsolver"
            solver_dir = "/home/hjpc/.ros/FORCESNLPsolver"
            self.N = N
            self.dt = dt
            self.load_model()
            if(model_build):
                self.model_build()            
            else:
                self.solver = forcespro.nlp.Solver.from_directory(solver_dir)                        
        except:            
            print("prebuilt solver not found")            
            self.model_build()  
        
        self.x0i = np.array([0.,0.,-1.5,1.5,1.,np.pi/4.])
        self.x0 = np.transpose(np.tile(self.x0i, (1, self.model.N)))
        self.xinit = np.transpose(np.array([-2.,0.,0.,np.deg2rad(90)]))
        self.problem = {"x0": self.x0,
            "xinit": self.xinit}
    
    def traversability_cost(self,z):
        A = np.zeros(1e3)
        A[2] = 1e5        
        cost =0.0 
        if z[2] > 1.0:
            cost = 0.1        
        return cost 

    def running_obj(self,z,p):
        # return 100 * casadi.fabs(z[2] -p[0]) + 100 * casadi.fabs(z[3] - p[1]) + 0.1 * z[0]**2 + 0.01 * z[1]**2+self.traversability_cost(z)
        return 0.1 * z[0]**2 + 0.01 * z[1]**2
    def terminal_obj(self,z,p):
        return 200 * casadi.fabs(z[2] -p[0]) + 200 * casadi.fabs(z[3] - p[1]) + 0.2 * z[0]**2 + 0.02 * z[1]**2

    def setState(self,x_np_array):
        self.xinit = np.transpose(x_np_array)
        self.problem["xinit"] = self.xinit

    def setParam(self,params_np_array):
        params_np_array = np.array([2.5, 2.5])        
        self.problem["all_parameters"] = np.transpose(np.tile(params_np_array,(1,self.model.N)))

    def load_model(self):
        self.model = forcespro.nlp.SymbolicModel()
        self.model.N = self.N # horizon length
        self.model.nvar = 6  # number of variables
        self.model.neq = 4  # number of equality constraints
        self.model.nh = 1  # number of inequality constraint functions
        self.model.npar = 4 # number of runtime parameters    
        # Objective function        
        self.model.objective = self.running_obj #lambda z: 100 * casadi.fabs(z[2] -5.0) \
                                   # + 100 * casadi.fabs(z[3] - 5.0) \
                                   # + 0.1 * z[0]**2 + 0.01 * z[1]**2
        self.model.objective = self.terminal_obj
        
        # We use an explicit RK4 integrator here to discretize continuous dynamics
        integrator_stepsize = self.dt
        self.model.eq = lambda z: forcespro.nlp.integrate(self.continuous_dynamics, z[2:6], z[0:2],
                                                    integrator=forcespro.nlp.integrators.RK4,
                                                    stepsize=integrator_stepsize)
        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        self.model.E = np.concatenate([np.zeros((4,2)), np.eye(4)], axis=1)

        # Inequality constraints
        # Simple bounds
        #  upper/lower variable bounds lb <= z <= ub
        #                     inputs                 |  states
        #                     F          phi                x            y     v             theta        delta
        # self.model.lb = np.array([-5.,  np.deg2rad(-40.),  -np.inf,   -np.inf,   -np.inf,  -np.inf, -0.38])
        # self.model.ub = np.array([+0.5,  np.deg2rad(+40.),   np.inf,   np.inf,    np.inf,    np.inf,  0.38])
        self.model.lb = np.array([-3.,  np.deg2rad(-25.),  -np.inf,   -np.inf,   -np.inf,  -np.inf])
        self.model.ub = np.array([+1.0,  np.deg2rad(+25.),   np.inf,   np.inf,    np.inf,    np.inf])
        # General (differentiable) nonlinear inequalities hl <= h(z,p) <= hu
        self.model.ineq = lambda z,p:  casadi.vertcat((z[2] -p[2]) ** 2 + (z[3] - p[3]) ** 2)
        # Upper/lower bounds for inequalities
        self.model.hu = np.array([+np.inf])
        self.model.hl = np.array([1.0**2])
        # Initial condition on vehicle states x
        self.model.xinitidx = range(2,6) # use this to specify on which variables initial conditions
       
        

    def model_build(self):
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 400     # Maximum number of iterations
        codeoptions.printlevel = 0  
        codeoptions.optlevel = 0    # 0 no optimization, 1 optimize for size, 
        #                             2 optimize for speed, 3 optimize for size & speed
        # codeoptions.cleanup = False
        codeoptions.nlp.hessian_approximation = 'bfgs'
        # codeoptions.solvemethod = 'SQP_NLP' # choose the solver method Sequential 
        codeoptions.nlp.bfgs_init = 3.0*np.identity(6) # initialization of the hessian
        #                             approximation
        codeoptions.noVariableElimination = 1.               
        # Creates code for symbolic model formulation given above, then contacts 
        # server to generate new solver
        self.solver = self.model.generate_solver(options=codeoptions)
        return self.model, self.solver

    def continuous_dynamics(self,x, u):
        """Defines dynamics of the car, i.e. equality constraints.
        parameters:
        state x = [xPos,yPos,v,theta,delta]
        input u = [F,phi]
        """
        # set physical constants
        l_r = 0.45 # distance rear wheels to center of gravitiy of the car
        l_f = 0.45 # distance front wheels to center of gravitiy of the car
        m = 20.5   # mass of the car

        # set parameters
        # beta = casadi.arctan(l_r/(l_f + l_r) * casadi.tan(x[4]))
        beta = casadi.arctan(l_r/(l_f + l_r) * casadi.tan(u[1]))

        # calculate dx/dt
        return casadi.vertcat(  x[2] * casadi.cos(x[3] + beta),  # dxPos/dt = v*cos(theta+beta)
                                x[2] * casadi.sin(x[3] + beta),  # dyPos/dt = v*sin(theta+beta)
                                u[0] / m,                        # dv/dt = F/m
                                x[2]/l_r * casadi.sin(beta))#,     # dtheta/dt = v/l_r*sin(beta)
                                # u[1])                           # ddelta/dt = phi
