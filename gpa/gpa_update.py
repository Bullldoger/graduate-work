"""
    Contains all needed classes
"""

import math
from copy import deepcopy

import networkx as nx
import sympy as sm
from sympy.solvers import solve_poly_system

import json


class BadInputParamsException(Exception):
    """
        Exception for controlling absent params
    """
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        if 'message' in kwargs:
            self.message = kwargs['message']
        else:
            self.message = 'unknown param'


class GPA(nx.DiGraph):
    """
        Represent a simple transport system
    """
    def __init__(self, params=None, **attr):
        """

        :param params: system configuration of gpa
        :type params: dict()

        params(example):
            {
              "approximation": {
                "p2": 1,
                "x1": 1,
                "x2": 1,
                "x3": 1
              },
              "edges": {
                "count": 3,
                "edges_info": {
                  "1": {
                    "A": 0.1,
                    "u": "1",
                    "v": "2",
                    "x": null
                  },
                  "2": {
                    "A": 0.1,
                    "u": "2",
                    "v": "3",
                    "x": null
                  },
                  "3": {
                    "A": 0.1,
                    "u": "2",
                    "v": "4",
                    "x": null
                  }
                }
              },
              "nodes": {
                "count": 4,
                "nodes_info": {
                  "1": {
                    "P": {
                      "P": 8,
                      "P_max": 100,
                      "P_min": 0
                    },
                    "Q": {
                      "Q": null,
                      "Q_max": 100,
                      "Q_min": 0
                    },
                    "name": "1"
                  },
                  "2": {
                    "P": {
                      "P": null,
                      "P_max": 100,
                      "P_min": 0
                    },
                    "Q": {
                      "Q": 0,
                      "Q_max": 100,
                      "Q_min": 0
                    },
                    "name": "2"
                  },
                  "3": {
                    "P": {
                      "P": 5,
                      "P_max": 100,
                      "P_min": 0
                    },
                    "Q": {
                      "Q": null,
                      "Q_max": 100,
                      "Q_min": 0
                    },
                    "name": "3"
                  },
                  "4": {
                    "P": {
                      "P": 5,
                      "P_max": 100,
                      "P_min": 0
                    },
                    "Q": {
                      "Q": null,
                      "Q_max": 100,
                      "Q_min": 0
                    },
                    "name": "4"
                  }
                }
              }
            }

        """
        super().__init__(**attr)
        if params is None:
            raise BadInputParamsException(message='Пустые входные параметры')

        self.n_v = params['nodes']['count']
        self.n_e = params['edges']['count']

        self.jacobi_matrixes = dict()

        self.sense_calculate_q = None
        self.sense_calculate_p = None
        self.current_sort_p = None
        self.current_sort_q = None

        self.d_approximation = None

        self.maxwell_matrix = None
        self.inverse_maxwell_matrix = None
        self.maxwell_matrix_p_q = None
        self.maxwell_matrix_q_p = None
        self.maxwell_matrix_p_p = None

        self.d_p_of_all_nodes = list()
        self.d_q_of_all_nodes = list()
        self.all_p_of_nodes = list()
        self.all_q_of_nodes = list()

        self.p_var = list()
        self.p_fix = list()
        self.q_var = list()
        self.q_fix = list()
        self.dp_variable = list()
        self.dq_variable = list()
        self.last_approximation = list()
        self.all_x_of_edges = list()
        self.incidence_matrix_with_known_q = list()
        self.d_of_all_known_q_of_nodes = list()
        self.incidence_matrix_with_known_p = list()
        self.symbols_of_all_q = list()
        self.incidence_matrix = list()

        self.last_approximation = None
        self.known_p_nodes = list()
        self.known_q_nodes = list()
        self.nodes_numeration = dict()
        self.variables = list()
        self.known_x_edges = list()
        self.internal_nodes = list()
        self.base_approximation = dict()
        self.approximations = list()

        self.q_to_x_mapping = None

        self._read_params(params=params)
        self._init_matrixes()
        self._init_equations()
        self._init_jacobi_matrixes()

    def _read_params(self, params=None):
        """

        :param params:
        :return:
        """

        assert params is not None

        if 'nodes' in params:
            self._read_nodes(params['nodes'].get('nodes_info'))

        if 'edges' in params:
            self._read_edges(params['edges'].get('edges_info'))

        self.known_p_nodes_indexes = [int(node) - 1 for node in self.known_p_nodes]
        self.known_q_nodes_indexes = [int(node) - 1 for node in self.known_q_nodes]

    def _read_nodes(self, nodes_info=None):
        """

        :param nodes_info: dictionary nodes attributes declaration and all values
        :type nodes_info: dict()

        node_part(example):
        {
            "count": 4,
            "nodes_info": {
              "1": {
                "P": {
                  "P": 8,
                  "P_max": 100,
                  "P_min": 0
                },
                "Q": {
                  "Q": null,
                  "Q_max": 100,
                  "Q_min": 0
                },
                "name": "1"
              },
              "2": {
                "P": {
                  "P": null,
                  "P_max": 100,
                  "P_min": 0
                },
                "Q": {
                  "Q": 0,
                  "Q_max": 100,
                  "Q_min": 0
                },
                "name": "2"
              },
              "3": {
                "P": {
                  "P": 5,
                  "P_max": 100,
                  "P_min": 0
                },
                "Q": {
                  "Q": null,
                  "Q_max": 100,
                  "Q_min": 0
                },
                "name": "3"
              },
              "4": {
                "P": {
                  "P": 5,
                  "P_max": 100,
                  "P_min": 0
                },
                "Q": {
                  "Q": null,
                  "Q_max": 100,
                  "Q_min": 0
                },
                "name": "4"
              }
        }

        :return:
        """
        if nodes_info is None:
            raise BadInputParamsException(message='Отсутствует информация по вершинам')

        assert isinstance(nodes_info, dict)

        for node_index, node in enumerate(sorted(nodes_info.keys(),
                                                 key=lambda node_number: node_number)):
            node_info = nodes_info[node]

            name = node_info['name']
            p_info = node_info['P']
            q_info = node_info['Q']

            p_of_node = p_info['P']
            p_max_of_node = p_info['P_max']
            p_min_of_node = p_info['P_min']

            q_of_node = q_info['Q']
            q_max_of_node = q_info['Q_max']
            q_min_of_node = q_info['Q_min']

            p_variable, q_variable = sm.symbols('p' + str(node_index + 1)), sm.symbols('q' + str(node_index + 1))
            self.dp_variable, self.dq_variable = sm.symbols('dp' + str(node_index + 1)), sm.symbols('dq' + str(node_index + 1))

            d_p = deepcopy(self.dp_variable)
            d_q = deepcopy(self.dq_variable)

            if p_of_node is None:
                p_of_node = deepcopy(p_variable)
                self.variables.append(deepcopy(p_of_node))
            else:
                self.known_p_nodes.append(node)
                self.base_approximation[str(p_variable)] = p_of_node

            if q_of_node is None:
                q_of_node = deepcopy(q_variable)
            else:
                self.known_q_nodes.append(node)
                self.base_approximation[str(q_variable)] = q_of_node

            self.add_node(node, name=name, P=p_of_node, Q=q_of_node, P_MAX=p_max_of_node,
                          P_MIN=p_min_of_node, Q_MAX=q_max_of_node, Q_MIN=q_min_of_node,
                          variable_p=p_variable, variable_q=q_variable, DP=d_p, DQ=d_q)
            self.nodes_numeration[node] = node_index

            self.d_p_of_all_nodes.append(d_p)
            self.d_q_of_all_nodes.append(d_q)

            self.all_p_of_nodes.append(p_variable)
            self.all_q_of_nodes.append(q_variable)

    def _read_edges(self, edge_part=None):
        """

        :param edge_part:
        :type edge_part: dict()

        edge_part(example):
        {
        "count": 3,
        "edges_info": {
            "1": {
                "A": 0.1,
                "u": "1",
                "v": "2",
                "x": null
                },
            "2": {
                "A": 0.1,
                "u": "2",
                "v": "3",
                "x": null
                },
            "3": {
                "A": 0.1,
                "u": "2",
                "v": "4",
                "x": null
            }
        }

        :return:
        """
        if edge_part is None:
            raise BadInputParamsException(message='Отсутствует информация для ребер')

        assert isinstance(edge_part, dict)

        self.q_to_x_mapping = {
            'to': {

            },
            'from': {

            }
        }

        for edge_index, edge in enumerate(edge_part.keys()):
            edge_info = edge_part[edge]

            u_of_edge = edge_info['u']
            v_of_edge = edge_info['v']
            a_of_edge = edge_info['A']
            x_of_edge = edge_info['x']

            node_u_info = self.nodes[u_of_edge]
            node_v_info = self.nodes[v_of_edge]

            self.q_to_x_mapping['to'][(u_of_edge, v_of_edge)] = node_v_info['variable_q']
            self.q_to_x_mapping['from'][(u_of_edge, v_of_edge)] = node_u_info['variable_q']

            variable_x = sm.symbols('x' + str(edge_index + 1))

            if x_of_edge is None:
                x_of_edge = deepcopy(variable_x)
                self.variables.append(deepcopy(x_of_edge))
            else:
                self.known_x_edges.append(edge)
                self.base_approximation[str(variable_x)] = x_of_edge

            self.add_edge(u_of_edge=u_of_edge, v_of_edge=v_of_edge,
                          A=a_of_edge, X=x_of_edge, variable_x=variable_x)

    def _init_matrixes(self):
        """

        Initializations of base matrix

        :return:
        """
        self.all_x_of_edges = sm.zeros(rows=self.n_e, cols=1)

        for edge_index, (edge, edge_info) in enumerate(sorted(self.edges.items(),
                                                              key=
                                                              lambda edge_sort_key: int(edge_sort_key[0][0]))):
            edge_index = list(self.edges()).index(edge)
            self.all_x_of_edges[edge_index] = edge_info['variable_x']

        self.incidence_matrix = nx.incidence_matrix(self, oriented=True).todense()
        self.symbols_of_all_q = sm.Matrix(deepcopy(self.all_q_of_nodes))
        self.all_q_of_nodes = self.incidence_matrix * self.all_x_of_edges
        self.incidence_matrix_with_known_q = deepcopy(self.incidence_matrix)
        self.incidence_matrix_with_known_p = deepcopy(self.incidence_matrix)

        for row in range(self.incidence_matrix_with_known_q.shape[0]):
            for col in range(self.incidence_matrix_with_known_q.shape[1]):
                if self.incidence_matrix_with_known_q[row, col] == -1:
                    self.incidence_matrix_with_known_q[row, col] = 0

                if self.incidence_matrix_with_known_p[row, col] == 1:
                    self.incidence_matrix_with_known_p[row, col] = 0

        self.d_of_all_known_q_of_nodes = sm.zeros(len(self.edges()), 1)
        self.d_of_all_known_p_of_nodes = sm.zeros(len(self.edges()), 1)

        for equation_index, (edge, edge_info) in enumerate(sorted(self.edges.items(),
                                                                  key=
                                                                  lambda edge_sort_key: edge_sort_key[0][0])):
            u_of_node = edge[0]
            v_of_node = edge[1]

            p_s = self.nodes[u_of_node]['variable_p']
            p_f = self.nodes[v_of_node]['variable_p']
            a_of_node = edge_info['A']

            equation_of_edge = sm.sqrt((p_s ** 2 - p_f ** 2) / a_of_node)

            self.d_of_all_known_q_of_nodes[equation_index] = sm.diff(equation_of_edge, p_s)
            self.d_of_all_known_p_of_nodes[equation_index] = -sm.diff(equation_of_edge, p_f)

        self.d_of_all_known_q_of_nodes = sm.diag(*self.d_of_all_known_q_of_nodes)
        self.d_of_all_known_p_of_nodes = sm.diag(*self.d_of_all_known_p_of_nodes)
        self.all_p_of_nodes = sm.Matrix(self.all_p_of_nodes)
        self.all_q_of_nodes = sm.Matrix(self.all_q_of_nodes)
        self.d_p_of_all_nodes = sm.Matrix(self.d_p_of_all_nodes)
        self.d_q_of_all_nodes = sm.Matrix(self.d_q_of_all_nodes)

    def _init_equations(self):
        """

        Initializations of equations

        :return:
        """
        equations = list()

        for u_of_edge, v_of_edge in self.edges():
            p_s = self.nodes[u_of_edge]['variable_p']
            p_f = self.nodes[v_of_edge]['variable_p']

            x_of_edge = self.edges[(u_of_edge, v_of_edge)]['variable_x']
            a_of_edge = self.edges[(u_of_edge, v_of_edge)]['A']

            equation = p_s ** 2 - p_f ** 2 - a_of_edge * x_of_edge ** 2

            equations.append(equation)

        x_equations = self.incidence_matrix * self.all_x_of_edges

        for internal_node in self.known_q_nodes:
            internal_node_index = list(self.nodes()).index(internal_node)
            equation = x_equations[internal_node_index] - self.nodes[internal_node]['Q']
            equations.append(equation)

        self.equations = sm.Matrix(equations)

    def _init_jacobi_matrixes(self):
        """

        Initializations all jacobians

        :return:
        """
        self.jacobi = sm.zeros(self.equations.shape[0], len(self.variables))

        for equations_index, equation in enumerate(self.equations):
            for variable_index, variable in enumerate(self.variables):
                pde_of_all_variables = sm.diff(equation, variable)
                self.jacobi[equations_index, variable_index] = pde_of_all_variables

        for variable_index, variable in enumerate(self.variables):
            jacobi_variable_matrix = deepcopy(self.jacobi)
            jacobi_variable_matrix[:, variable_index] = self.equations
            self.jacobi_matrixes[variable] = jacobi_variable_matrix

    def _solve_step(self, approximation):
        """

        :param approximation: approximation of current point
        :type approximation: dict()

        approximation(example):
        {
            "p2": 1,
            "x1": 1,
            "x2": 1,
            "x3": 1
        }

        :return: new approximation
        """
        solve_system = dict()
        for variable, jacobi_variable in self.jacobi_matrixes.items():
            jacobi = deepcopy(self.jacobi)

            jacobi = jacobi.subs(approximation.items()).det()
            jacobi_variable = jacobi_variable.subs(approximation.items()).det()

            solve_system[variable] = jacobi_variable.evalf() / jacobi.evalf()

        for variable, value in solve_system.items():
            approximation[str(variable)] = approximation[str(variable)] - value

        return approximation

    def solve_equations(self, base_approximation=None,
                        start_approximation=None, eps=0.001, iterations=25):
        """

        :param base_approximation:
        :param start_approximation: first approximation for searching system decision
        :type start_approximation: dict()
        :param eps: accuracy
        :type eps: float
        :param iterations: maximum of iterations for searching dicision
        :type iterations: int

        start_approximation(example):
        {
            "p2": 1,
            "x1": 1,
            "x2": 1,
            "x3": 1
        }


        :return: searched dicision
        """
        if start_approximation is None:
            raise KeyError('Начальное приближение не задано!')

        if base_approximation is None:
            base_approximation = deepcopy(self.base_approximation)

        start_approximation_vars = list(start_approximation.keys())

        for var in self.variables:
            if str(var) not in start_approximation_vars:
                raise KeyError('Не задано начальное значение для ' + str(var))

        approximation = deepcopy(start_approximation)
        approximation.update(deepcopy(base_approximation))
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

        self.last_approximation = deepcopy(approximation)
        self.approximations.append(deepcopy(approximation))

        return self

    def subs_solved_variables(self, approximation=None):
        """

        :param approximation: approximation for subs to equations
        :type approximation: dict
        approximation(example):
        {
            "p2": 1,
            "x1": 1,
            "x2": 1,
            "x3": 1
        }


        :return:
        """

        for node in self.nodes():
            node_info = self.nodes[node]
            for var, value in approximation.items():
                if str(node_info['P']) == var:
                    node_info['P'] = value

        for edge in self.edges():
            edge_info = self.edges[edge]
            for var, value in approximation.items():
                if str(edge_info['X']) == var:
                    edge_info['X'] = value

        return self

    def construct_sense_matrix(self, base_approximation=None, d_approximation=None):
        """

        Constructing all sensitivities matrixes
        :return:

        """
        if base_approximation is None:
            base_approximation = deepcopy(self.last_approximation)

        assert base_approximation is not None

        self.p_var = self.d_p_of_all_nodes[self.known_q_nodes_indexes, :]
        self.p_fix = self.d_p_of_all_nodes[self.known_p_nodes_indexes, :]
        self.q_var = self.d_q_of_all_nodes[self.known_p_nodes_indexes, :]
        self.q_fix = self.d_q_of_all_nodes[self.known_q_nodes_indexes, :]

        a_at_current_order = deepcopy(self.incidence_matrix[self.known_q_nodes_indexes + self.known_p_nodes_indexes, :])

        a_q = self.incidence_matrix[self.known_q_nodes_indexes, :]
        a_p = self.incidence_matrix[self.known_p_nodes_indexes, :]

        a_f_q = self.incidence_matrix_with_known_q[self.known_q_nodes_indexes, :]
        a_f_p = self.incidence_matrix_with_known_q[self.known_p_nodes_indexes, :]

        a_l_q = self.incidence_matrix_with_known_p[self.known_q_nodes_indexes, :]
        a_l_p = self.incidence_matrix_with_known_p[self.known_p_nodes_indexes, :]

        d_f = self.d_of_all_known_q_of_nodes.subs(base_approximation.items())
        d_l = self.d_of_all_known_p_of_nodes.subs(base_approximation.items())

        self.maxwell_matrix = a_q * (d_f * a_f_q.transpose() + d_l * a_l_q.transpose())
        self.inverse_maxwell_matrix = deepcopy(self.maxwell_matrix).inv()
        self.maxwell_matrix_p_q = a_p * (d_f * a_f_q.transpose() + d_l * a_l_q.transpose())
        self.maxwell_matrix_q_p = a_q * (d_f * a_f_p.transpose() + d_l * a_l_p.transpose())
        self.maxwell_matrix_p_p = a_p * (d_f * a_f_p.transpose() + d_l * a_l_p.transpose())

        self.p_var = self.inverse_maxwell_matrix * self.q_fix - self.inverse_maxwell_matrix * self.maxwell_matrix_q_p * self.p_fix
        self.q_var = self.maxwell_matrix_p_q * self.inverse_maxwell_matrix * self.q_fix + (self.maxwell_matrix_p_p - self.maxwell_matrix_p_q * self.inverse_maxwell_matrix * self.maxwell_matrix_q_p) * self.p_fix

        if d_approximation is not None:
            all_approximation = deepcopy(d_approximation)

            self.p_var = self.p_var.subs(all_approximation.items())
            self.q_var = self.q_var.subs(all_approximation.items())

            self.sense_calculate_q = (a_at_current_order * self.all_x_of_edges).subs(base_approximation.items()) + \
                                     sm.Matrix([self.q_fix, self.q_var]).subs(d_approximation.items())
            self.sense_calculate_p = self.all_p_of_nodes[self.known_p_nodes_indexes + self.known_q_nodes_indexes, :].subs(base_approximation.items()) + \
                                     sm.Matrix([self.p_fix, self.p_var]).subs(d_approximation.items())

            self.current_sort_p = self.all_p_of_nodes[self.known_p_nodes_indexes + self.known_q_nodes_indexes, :]
            self.current_sort_q = self.symbols_of_all_q[self.known_q_nodes_indexes + self.known_p_nodes_indexes, :]

            self.d_approximation = dict()

            for variable, value in zip(self.current_sort_p, self.sense_calculate_p):
                self.d_approximation[str(variable)] = value

            print(self.current_sort_q)
            print(self.sense_calculate_q)

            x_solved_system = self.incidence_matrix * self.all_x_of_edges - self.sense_calculate_q

            print(x_solved_system)

            x_solved_system = x_solved_system[range(len(self.variables))[:-1], :]

            print(x_solved_system)

            x_solved_system = solve_poly_system(x_solved_system,
                                                self.all_x_of_edges.atoms(sm.Symbol))

            x_solved_system = x_solved_system[0]
            x_solved_system = {str(sym): value for value, sym in zip(x_solved_system,
                                                                     self.all_x_of_edges.atoms(sm.Symbol))}

            self.d_approximation.update(x_solved_system)
            self.approximations.append(self.d_approximation)

        return self
