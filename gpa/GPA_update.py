import networkx as nx
import sympy as sm
import numpy

import math
from copy import deepcopy


class BadInputParamsException(Exception):
    def __init__(self, **kwargs):
        if 'message' in kwargs:
            self.message = kwargs['message']
        else:
            self.message = 'unknown param'


class GPA(nx.DiGraph):

    def __init__(self, params=None, **attr):
        super().__init__(**attr)
        if params is None:
            raise BadInputParamsException(message='Пустые входные параметры')

        self.n_v = params['nodes']['count']
        self.n_e = params['edges']['count']

        self.jacobi_matrixes = dict()

        self._read_params(params=params)
        self._init_matrixes()
        self._init_equations()
        self._init_jacobi_matrixes()

    def _read_params(self, params=None):

        self.known_p_nodes = list()
        self.known_q_nodes = list()
        self.nodes_numeration = dict()
        self.variables = list()
        self.known_x_edges = list()
        self.internal_nodes = list()
        self.base_approximation = dict()

        if 'nodes' in params:
            self._read_nodes(params['nodes'].get('nodes_info'))

        if 'edges' in params:
            self._read_edges(params['edges'].get('edges_info'))

    def _read_nodes(self, nodes_info=None):

        if nodes_info is None:
            raise BadInputParamsException(message='Отсутствует информация по вершинам')

        assert isinstance(nodes_info, dict)

        self.dP = list()
        self.dQ = list()

        self.P = list()
        self.Q = list()

        for node_index, node in enumerate(sorted(nodes_info.keys(), key=lambda node_number: node_number)):
            node_info = nodes_info[node]

            name = node_info['name']
            p_info = node_info['P']
            q_info = node_info['Q']

            p = p_info['P']
            p_max = p_info['P_max']
            p_min = p_info['P_min']

            q = q_info['Q']
            q_max = q_info['Q_max']
            q_min = q_info['Q_min']

            p_variable, q_variable = sm.symbols('p' + str(node_index + 1)), sm.symbols('q' + str(node_index + 1))
            self.dp_variable, self.dq_variable = sm.symbols('dp' + str(node_index + 1)), sm.symbols('dq' + str(node_index + 1))

            dp = deepcopy(self.dp_variable)
            dq = deepcopy(self.dq_variable)

            if p is None:
                p = deepcopy(p_variable)
                self.variables.append(deepcopy(p))
            else:
                self.known_p_nodes.append(node)
                self.base_approximation[str(p_variable)] = p

            if q is None:
                q = deepcopy(q_variable)
            else:
                self.known_q_nodes.append(node)
                self.base_approximation[str(q_variable)] = q

            self.add_node(node, name=name, P=p, Q=q, P_MAX=p_max,
                          P_MIN=p_min, Q_MAX=q_max, Q_MIN=q_min, variable_p=p_variable,
                          variable_q=q_variable, DP=dp, DQ=dq)
            self.nodes_numeration[node] = node_index

            self.dP.append(dp)
            self.dQ.append(dq)

            self.P.append(p_variable)
            self.Q.append(q_variable)

    def _read_edges(self, edge_part=None):

        if edge_part is None:
            raise BadInputParamsException(message='Отсутствует информация для ребер')

        assert isinstance(edge_part, dict)

        for edge_index, edge in enumerate(sorted(edge_part.keys(), key=lambda edge_number: edge_number)):
            edge_info = edge_part[edge]

            u = edge_info['u']
            v = edge_info['v']
            a = edge_info['A']
            x = edge_info['x']

            variable_x = sm.symbols('x' + str(edge_index + 1))

            if x is None:
                x = deepcopy(variable_x)
                self.variables.append(deepcopy(x))
            else:
                self.known_x_edges.append(edge)
                self.base_approximation[str(variable_x)] = x

            self.add_edge(u_of_edge=u, v_of_edge=v, A=a, X=x, variable_x=variable_x)

    def _init_matrixes(self):

        self.X = sm.zeros(rows=self.n_e, cols=1)

        for edge_index, (edge, edge_info) in enumerate(sorted(self.edges.items(), key=lambda edge: edge[0][0])):
            edge_index = list(self.edges()).index(edge)
            self.X[edge_index] = edge_info['variable_x']

        self.A = nx.incidence_matrix(self, oriented=True).todense()
        self.Q = self.A * self.X
        self.AF = deepcopy(self.A)
        self.AL = deepcopy(self.A)

        for row in range(self.AF.shape[0]):
            for col in range(self.AF.shape[1]):
                if self.AF[row, col] == -1:
                    self.AF[row, col] = 0

                if self.AL[row, col] == 1:
                    self.AL[row, col] = 0

        self.DF = sm.zeros(len(self.edges()), 1)
        self.DL = sm.zeros(len(self.edges()), 1)

        for equation_index, (edge, edge_info) in enumerate(sorted(self.edges.items(), key=lambda edge: edge[0][0])):
            u = edge[0]
            v = edge[1]

            p_s = self.nodes[u]['variable_p']
            p_f = self.nodes[v]['variable_p']
            a = edge_info['A']

            eq = sm.sqrt((p_s ** 2 - p_f ** 2) / a)

            self.DF[equation_index] = sm.diff(eq, p_s)
            self.DL[equation_index] = -sm.diff(eq, p_f)

        self.DF = sm.diag(*self.DF)
        self.DL = sm.diag(*self.DL)
        self.P = sm.Matrix(self.P)
        self.Q = sm.Matrix(self.Q)
        self.dP = sm.Matrix(self.dP)
        self.dQ = sm.Matrix(self.dQ)

    def _init_equations(self):

        equations = list()

        for u, v in self.edges():
            p_s = self.nodes[u]['variable_p']
            p_f = self.nodes[v]['variable_p']

            x = self.edges[(u, v)]['variable_x']
            a = self.edges[(u, v)]['A']

            eq = p_s ** 2 - p_f ** 2 - a * x ** 2

            equations.append(eq)

        x_equations = self.A * self.X

        for internal_node in self.known_q_nodes:
            internal_node_index = list(self.nodes()).index(internal_node)
            equation = x_equations[internal_node_index]
            equations.append(equation)

        self.equations = sm.Matrix(equations)

    def _init_jacobi_matrixes(self):
        self.jacobi = sm.zeros(self.equations.shape[0], len(self.variables))

        for equations_index, equation in enumerate(self.equations):
            for variable_index, variable in enumerate(self.variables):
                partial_differencial_equation_by_variable = sm.diff(equation, variable)
                self.jacobi[equations_index, variable_index] = partial_differencial_equation_by_variable

        for variable_index, variable in enumerate(self.variables):
            jacobi_variable_matrix = deepcopy(self.jacobi)
            jacobi_variable_matrix[:, variable_index] = self.equations
            self.jacobi_matrixes[variable] = jacobi_variable_matrix

    def _solve_step(self, approximation):

        solve_system = dict()
        for variable, jacobi_variable in self.jacobi_matrixes.items():
            jacobi = deepcopy(self.jacobi)

            jacobi = jacobi.subs(approximation.items()).det()
            jacobi_variable = jacobi_variable.subs(approximation.items()).det()

            solve_system[variable] = jacobi_variable.evalf() / jacobi.evalf()

        for variable, value in solve_system.items():
            approximation[str(variable)] = approximation[str(variable)] - value

        return approximation

    def solve_equations(self, base_approximation=None, start_approximation=None, eps=0.001, iterations=25):

        if start_approximation is None:
            raise KeyError('Начальное приближение не задано!')

        if base_approximation is None:
            base_approximation = deepcopy(self.base_approximation)

        start_approximation_vars = list(start_approximation.keys())

        for var in self.variables:
            if str(var) not in start_approximation_vars:
                raise KeyError('Не задано начальное значение для ' + str(var))

        approximation = start_approximation
        approximation.update(base_approximation)
        errors = list()

        for _ in range(iterations):

            approximation = self._solve_step(approximation)
            approximation.update(base_approximation)
            error = deepcopy(self.equations).subs(approximation.items()).norm()
            errors.append(error)

            if error < eps:
                break

        points = -int(math.log10(eps))

        for var in approximation:
            approximation[var] = round(approximation[var], points)

        return approximation
