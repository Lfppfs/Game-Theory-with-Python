# Part I

# These are the packages needed
import getpass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
from os import chdir, getcwd, listdir, mkdir
from scipy import integrate
from sympy import diff, functions, init_printing, Matrix,\
    numbers, Poly, solve, symbols, sympify
from sympy.solvers.inequalities import solve_poly_inequality

username = getpass.getuser()  # this will be useful for saving a plot as a figure
# init_printing(use_latex='mathjax') # this is simply for aesthetics,
# more useful when running the code on IPython


class System():
    '''
    class System()
        Base class for system objects, i.e., class for setting and analyzing a
        differential equation system based on a Game Theory matrix.
        The t variable may be set to a numerical value using t_value,
        or analyzed analytically by specifying t = \'set\' and t_value = None.

        Methods defined here:

        solve(self, game = None, t = None)
            Solves the system using sympy's solve function;
            sets the Jacobian matrix for the system using
            sympy's Matrix.jacobian; and evaluates the Jacobian at each
            solution point.

            Returns
            -------
            solutions: list containing the solutions of the system
            jac: sympy.matrices.dense.MutableDenseMatrix,
            Jacobian matrix of the system
            jac_at_point: list containing the Jacobian matrix of the system
            evaluated at each solution point

        eigs(self, save = False)
            Calculates the eigenvalues of the Jacobian matrix at each
            solution point; since looking at the eigenvalues directly isn't
            trictly necessary for determining the stability of the
            equilibrium points, this is a separate method from solve().

            Returns
            -------
            eigenvals: eigenvalues of the system, appended to jac_at_point.

    '''

    def __init__(self, variables, game='PD', t=None, t_value=None):

        self.variables = list(map(symbols, variables))
        self.t = symbols('t')

#         these if-else statements allow changing the game
        if game == 'PD':
            self.matrix = np.array([[3, 0], [5, 1]], dtype=float)
        elif game == 'HD':
            self.matrix = np.array([[3, 1], [4, 0]], dtype=float)
        else:
            self.matrix = np.array(game, dtype=float)

#         fitness functions
        self.fit_c = self.variables[0] * self.matrix[0, 0] +\
            (1 - self.variables[0]) * self.matrix[0, 1]

        self.fit_d = self.variables[0] * self.matrix[1, 0] +\
            (1 - self.variables[0]) * self.matrix[1, 1]

#         these lines allow working with a free parameter in the
#         entry [1,0] of the matrix
        if t == 'set':
            self.fit_d = (self.variables[0]) * self.t.subs(self.t, t_value) +\
                (1 - self.variables[0]) * self.matrix[1, 1]
        elif t != 'set' and t != None:
            raise ValueError('t must be either \'set\' or NoneType')

#         replicator equation
        self.replicator = self.variables[0] * \
            (1 - self.variables[0]) * (self.fit_c - self.fit_d)

    def __call__(self, coop_frequency, def_frequency):
        return self.replicator.subs([(self.variables[0], coop_frequency),
                                     (self.variables[1], def_frequency)])

    def solve(self, save=False, directory=None, filename=None):
        '''
        solve(self, game = None, t = None)
            Solves the system using sympy's solve function;
            sets the Jacobian matrix for the system using
            sympy's Matrix.jacobian; and evaluates the Jacobian at each
            solution point.

            Returns
            -------
            solutions: list containing the solutions of the system
            jac: sympy.matrices.dense.MutableDenseMatrix,
            Jacobian matrix of the system
            jac_at_point: list containing the Jacobian matrix of the system
            evaluated at each solution point
        '''

        self.solutions = solve(self.replicator, self.variables, dict=True)

        self.functions = Matrix([self.replicator])
        self.variables = Matrix([self.variables])
        self.jac = Matrix.jacobian(self.functions, self.variables)

#         evaluating the jacobian at every solution point and appending
#         these to jac_at_point
        self.jac_at_point = []
        for i in self.solutions:
            self.jac_at_point.append(self.jac.subs(
                [(list(i.items())[0][0], list(i.items())[0][1])]))
        return

    def eigs(self, save=False, directory=None, filename=None):
        '''
        eigs(self, save = False)
            Calculates the eigenvalues of the Jacobian matrix at each
            solution point; since looking at the eigenvalues directly isn't
            trictly necessary for determining the stability of the
            equilibrium points, this is a separate method from solve().

            Returns
            -------
            eigenvals: eigenvalues of the system, appended to jac_at_point.
        '''

#         calculating the eigenvalues of each solution and appending
#         these to eigenvals
        self.eigenvals = []
        for i in self.jac_at_point:
            self.eigenvals.append(i.eigenvals())
        return

#         saving the eigenvalues to a txt file
        if save:
            with open(directory + filename + '.txt', '+w') as f:
                f.write('\n\neigenvalues:\n\n')
                for index, e in enumerate(self.eigenvals):
                    f.write('{}) {}\n\n'.format(index, e))


# A quick example
PD_game = System(['x'])

# Game matrix
print(PD_game.matrix)

# Fitness functions
print(PD_game.fit_c, PD_game.fit_d)

# Replicator equation:
print(PD_game.replicator)

# solve method solves the system and defines the attributes solutions, jac and jac_at_point
PD_game.solve()
print(PD_game.solutions, PD_game.jac, PD_game.jac_at_point)


# eigs method calculates the eigenvalues of the system
PD_game.eigs()
print(PD_game.eigenvals)

# t = 'set' makes the [1,0] position of the matrix a free parameter
PD_game = System(variables=['x'], t='set')
PD_game.solve()
PD_game.eigs()
print(PD_game.solutions, PD_game.eigenvals)

# Part II


class FES(System):

    def __init__(self, variables, game='PD', theta_value=None):
        self.variables = list(map(symbols, variables))
        self.theta = symbols('theta')

        #         these if-else statements allow changing the game
        if game == 'PD':
            self.matrix = np.array([[3, 0], [5, 1]], dtype=float)
        elif game == 'HD':
            self.matrix = np.array([[3, 1], [4, 0]], dtype=float)
        else:
            self.matrix = np.array(game, dtype=float)

#         fitness functions
        self.fit_c =\
            self.variables[0] *\
            (self.matrix[1, 0] - (self.matrix[1, 0] -
                                  self.matrix[0, 0]) * self.variables[1]) +\
            (1 - self.variables[0]) *\
            (self.matrix[1, 1] - (self.matrix[1, 1] -
                                  self.matrix[0, 1]) * self.variables[1])

        self.fit_d =\
            self.variables[0] *\
            (self.matrix[0, 0] + (self.matrix[1, 0] -
                                  self.matrix[0, 0]) * self.variables[1]) +\
            (1 - self.variables[0]) *\
            (self.matrix[0, 1] + (self.matrix[1, 1] -
                                  self.matrix[0, 1]) * self.variables[1])

#         resource availability function
        self.env = self.theta.subs(
            self.theta, theta_value) * self.variables[0] -\
            (1 - self.variables[0])

#         replicator equations
        self.x_rep = self.variables[0] * \
            (1 - self.variables[0]) * (self.fit_c - self.fit_d)
        self.n_rep = self.variables[1] * (1 - self.variables[1]) * self.env

        self.replicator = [self.x_rep, self.n_rep]

    def __call__(self, coop_frequency, def_frequency):
        return System.__call__(self, coop_frequency, def_frequency)

    def solve(self, save=False, directory=None, filename=None):
        return System.solve(self, save=False, directory=None, filename=None)

    def eigs(self, save=False, directory=None, filename=None):
        return System.eigs(self, save=False, directory=None, filename=None)


game = FES(['x', 'n'])
game.solve(), game.eigs()
game.solutions

# Solving eigenvalue number 6 for n when theta = 1.5
inequality = list(game.eigenvals[6].items())[1][0].subs([(game.theta, 1)])
solve_poly_inequality(Poly(inequality, game.variables[1], domain='R'), '<')

# Phase space and time series


def graph_data(model, frequencies, theta_val, t_points):
    '''
        graph_data(frequencies,theta_val,t_points)

            Calculates data for plotting a phase space a (with a phase space
            trajectory for a chosen initial point of the system) and a scatter
            plot of $(c,n) \times t$. Calls integrate.solve_ivp from the Scipy
            package and several functions from Numpy.

            Parameters
            ----------
            frequencies : list, initial frequencies of c and n

            theta_val : num, value of theta in the differential equation

            t_points : list, time points for which the function should
            be evaluated

            Returns
            -------
            integ_sol : bunch object with several fields, see the
            documentation for integrate.solve_ivp for more info
            c_grid, n_grid : numpy.ndarray, grids with arrow locations for
            a quiver plot
            c_dir, n_dir : numpy.ndarray, grids with arrow directions for
            a quiver plot
    '''

#     this function is a callable needed by integrate.solve_ivp(below)
    def derivs(t, freqs):
        return np.array([float(model.x_rep.subs([
            (model.variables[0], freqs[0]),
            (model.variables[1], freqs[1])])),
            float(model.n_rep.subs([
                (model.variables[0], freqs[0]),
                (model.variables[1], freqs[1]),
                (model.theta, theta_val)]))])

#     line below will solve the system numerically. See the documentation for
#     integrate.solve_ivp for more info
    integ_sol = integrate.solve_ivp(
        derivs, [0, 1000], [frequencies[0], frequencies[1]], t_eval=t_points)

#     grids below will be used to plot the phase space
#     See the examples of quiverplots at the Matplotlib tutorial for further
#     information at:
#     https://matplotlib.org/gallery/images_contours_and_fields/quiver_demo.html

#     grid for arrow locations
    n_grid,  x_grid = np.meshgrid(np.arange(0, 1.1, 0.1),
                                  np.arange(0, 1.1, 0.1))
    x_dir, n_dir, = x_grid.copy(), n_grid.copy()
#     grid for arrow directions
    for width in range(len(x_dir)):
        for length in range(len(n_dir)):
            x_dir[width, length], n_dir[width, length] =\
                model.x_rep.subs([
                    (model.variables[0], x_grid[width, length]),
                    (model.variables[1], n_grid[width, length])]),\
                model.n_rep.subs([
                    (model.variables[0], x_grid[width, length]),
                    (model.variables[1], n_grid[width, length]),
                    (model.theta, theta_val)])

    return integ_sol, x_grid, n_grid, x_dir, n_dir


#  Solving the system numerically for x_0 = 0.6 and n_0 = 0.2, with theta=1.5
solution = graph_data(game, [0.6, 0.2], 0.5, np.linspace(0, 200, 1000))
solution[0].success  # returns True if the integration was successfull

# Plotting the phase space and time series


def make_graphs(model=None, data=None, save=False,
                directory=os.path.join('/home', username, 'Desktop/graphs'),
                fig_name=''):

    plt.close('all')

#     some plot settings
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=[14, 5])
    gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.2, hspace=0.25)
    line_color = ['#f88b25', '#2590f8', '#1ac129']

    if data != None:
        model = graph_data([data[0][0], data[0][1]],
                           theta_val=data[1], t_points=data[2])

    solution_label = fr'Model solution starting from ' +\
        fr'$x_0={model[0].y[0][0]}, n_0={model[0].y[1][0]}$'

    axq = fig.add_subplot(gs[0, 0])
    # axq.quiver plots the phase space
    axq.quiver(model[1], model[2], model[3], model[4], alpha=0.35)
    # the line below plots the phase space trajectory
    axq_solution, = axq.plot(model[0].y[0], model[0].y[1],
                             linewidth=3, color=line_color[0],
                             label=solution_label)
    axq.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0., fontsize=12)

    axq.set_xlabel(fr"$x$", fontsize=15)
    axq.set_ylabel(fr"$n$", fontsize=15)

    # lines below plot the time evolution of the system
    ax_curr = fig.add_subplot(gs[0, 1])
    ax_curr.plot(model[0].t, model[0].y[0], marker='', markersize=8, mew=2,
                 color=line_color[1], linewidth=3, label='x')
    ax_curr.plot(model[0].t, model[0].y[1], marker='', markersize=8, mew=2,
                 color=line_color[2], linewidth=3, label='n')

    ticks = np.linspace(0, 1, 5).round(2)
    ax_curr.set_yticks(ticks)
    ax_curr.set_ylim(0, 1)
    ax_curr.set_xlim(-1., max(model[0].t))
    ax_curr.set_ylabel("Variables", fontsize=17)
    ax_curr.set_xlabel("Time", fontsize=15)
    ax_curr.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0., fontsize=15)

    if save:  # for saving the figure in a given directory with a given name
        save_obj = os.path.join(directory, fig_name)
        plt.savefig(save_obj, dpi=90)


make_graphs(model=solution,
            save=False, directory='/home/user/Desktop/',
            fig_name='game.png')
