"""
    Contain a class declaration with represent GPA system
"""

import math
from copy import deepcopy

import sympy as sm
import networkx as nx
import numpy
from numpy import dot


class GPA:
    """
        Represent GPA system
    """

    def __init__(self, params=None):
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
        self.gpa = nx.DiGraph()
        "attribute GPA.gpa GPA structure graph"

        self.N_v = params['nodes']['count']
        "attribute GPA.N_v - number of nodes"

        self.N_e = params['edges']['count']
        "attribute GPA.N_e - number of edges"

        self.p_var = self.p_fix = self.q_var = self.q_fix = self.dp_var = self.dq_var = None
        "attribute GPA.p_var - variables p"
        "attribute GPA.p_fix - fixed p"
        "attribute GPA.q_var - variables q"
        "attribute GPA.q_fix - fixed q"

        self.M_PP = self.M_PQ = self.M_QP = self.M = self.inv_M = None

        self.known_p_nodes = list()
        "attribute GPA.known_p_nodes - vertex order"

        self.known_q_nodes = list()
        "attribute GPA.known_q_nodes- edges order"

        self.unknown_p = list()
        "attribute GPA.unknown_p - names of p variables"

        self.unknown_q = list()
        "attribute GPA.unknown_q - names of q variables"

        self.base_approximation = dict()
        "attribute GPA.base_approximation - values of variables from params"

        self.find_approximation = None
        "attribute GPA.find_approximation  - founded variables"

        self.full_approximation = None

        self.matrix_rows_order = list()
        "attribute GPA.matrix_rows_order - rows order"

        self.nodes_info = params['nodes']['nodes_info']
        self.edges_info = params['edges']['edges_info']

        self.variables = list()
        "attribute GPA.variables - variables for searching"

        self.var_literals = list()
        "attribute GPA.var_literals - names of variables for searching"

        self._read_nodes(self.nodes_info)
        self._read_edges(self.edges_info)

        self._init_matrix()
        self._init_equations()
        self._init_jacobi_matrix()

        self.solving_errors = list()

    def _read_nodes(self, node_part=None):
        """

        :param node_part: dictionary nodes attributes declaration and all values
        :type node_part: dict()

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

        if node_part is None:
            node_part = {}
        self.nodes_numeration = dict()
        node_index = 0
        for node, value in sorted(node_part.items(), key=lambda x: x[0]):

            self.nodes_numeration[node] = node_index
            node_index += 1

            name = value['name']
            p_info = value['P']
            q_info = value['Q']

            p = p_info['P']
            p_max = p_info['P_max']
            p_min = p_info['P_min']

            q = q_info['Q']
            q_max = q_info['Q_max']
            q_min = q_info['Q_min']

            var_p = sm.symbols('p' + str(node))
            var_q = sm.symbols('q' + str(node))

            self.unknown_p.append(var_p)

            if p == None:
                p = sm.symbols('p' + str(node))
                self.variables.append(p)
                self.var_literals.append(str(p))
            else:
                self.base_approximation[str(var_p)] = p
                self.known_p_nodes.append(node)

            if q == None:
                q = sm.symbols('q' + str(node))
                self.unknown_q.append(q)
            else:
                self.base_approximation[str(var_q)] = q
                self.known_q_nodes.append(node)

            self.gpa.add_node(node, P=p, P_min=p_min,
                              P_max=p_max, Q=q, Q_min=q_min,
                              Q_max=q_max, name=name, var=var_p)

        self.nodes = list(self.gpa.nodes())

        for node in self.known_q_nodes + self.known_p_nodes:
            self.matrix_rows_order.append(self.nodes.index(node))

        tmp_p = list()
        for node in self.known_p_nodes:
            tmp_p.append(self.nodes.index(node))

        tmp_q = list()
        for node in self.known_q_nodes:
            tmp_q.append(self.nodes.index(node))

        self.known_p_nodes, self.known_q_nodes = tmp_p, tmp_q

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
        internal_nodes = numpy.zeros([2, self.N_v])

        for node_index, value in sorted(edge_part.items(), key=lambda sorting_param: int(sorting_param[0][0])):
            u = value['u']
            v = value['v']
            a = value['A']
            x = value['x']

            u_index = self.nodes_numeration[u]
            v_index = self.nodes_numeration[v]

            internal_nodes[0, u_index] += 1
            internal_nodes[1, v_index] += 1
            var_x = sm.symbols('x' + str(node_index))

            if not x:
                x = var_x
                self.variables.append(x)
                self.var_literals.append(str(x))

            self.gpa.add_edge(u_of_edge=u, v_of_edge=v, A=a, x=x, var=var_x)

        internal_nodes = numpy.logical_and(internal_nodes[0, :], internal_nodes[1, :])
        self.internal_nodes = set()
        for node_index, is_internal in enumerate(internal_nodes):
            if is_internal:
                self.internal_nodes.add(str(node_index + 1))

    def _init_matrix(self):
        """

        Initializations of base matrix

        :return:
        """
        self.X = sm.zeros(rows=self.N_e, cols=1)
        for edge in self.gpa.edges():
            edge_index = list(self.gpa.edges()).index(edge)
            edge_params = self.gpa.edges[edge]

            self.X[edge_index] = edge_params['x']

        self.X = numpy.array(self.X)
        self.A = numpy.array(nx.incidence_matrix(self.gpa, oriented=True).todense())
        self.Q = sm.Matrix(numpy.dot(self.A, self.X))

        self.P = sm.zeros(rows=self.N_v, cols=1)
        for node in self.gpa.nodes():
            node_index = list(self.gpa.nodes()).index(node)
            node_params = self.gpa.nodes[node]

            self.P[node_index] = node_params['var']

        self.X = sm.Matrix(self.X)
        self.Q = sm.Matrix(self.Q)
        self.A = sm.Matrix(self.A)

        self.AF = deepcopy(self.A)
        self.AL = deepcopy(self.A)

        for row in range(self.AF.shape[0]):
            for col in range(self.AF.shape[1]):
                if self.AF[row, col] == -1:
                    self.AF[row, col] = 0

        for row in range(self.AL.shape[0]):
            for col in range(self.AL.shape[1]):
                if self.AL[row, col] == 1:
                    self.AL[row, col] = 0

        self.DF = sm.zeros(len(self.gpa.edges()), 1)
        self.DL = sm.zeros(len(self.gpa.edges()), 1)

        for eq_index, edge in enumerate(self.gpa.edges()):
            u = edge[0]
            v = edge[1]

            p_s = self.gpa.nodes[u]['var']
            p_f = self.gpa.nodes[v]['var']
            a = self.gpa.edges[edge]['A']

            eq = sm.sqrt((p_s ** 2 - p_f ** 2) / a)

            self.DF[eq_index] = sm.diff(eq, p_s)
            self.DL[eq_index] = -sm.diff(eq, p_f)

        self.DF = sm.diag(*self.DF)
        self.DL = sm.diag(*self.DL)

    def _init_equations(self):
        """

        Initializations of equations

        :return:
        """
        edges = self.gpa.edges()
        nodes = self.gpa.nodes()

        self.equations = list()

        for u, v in sorted(self.gpa.edges(), key=lambda x: int(x[0])):
            p_s = nodes[u]['P']
            p_f = nodes[v]['P']
            q = edges[(u, v)]['x']
            a = edges[(u, v)]['A']

            eq = p_s ** 2 - p_f ** 2 - a * q ** 2

            self.equations.append(eq)

        for el in self.internal_nodes:
            eq_index = list(self.gpa.nodes()).index(el)
            eq = deepcopy(self.Q[eq_index])

            self.equations.append(eq)
            self.Q[eq_index] = self.gpa.nodes[el]['Q']
        self.equations = sm.Matrix(self.equations)

    def _init_jacobi_matrix(self):
        """

        Initializations all jacobians

        :return:
        """
        self.jacobi_matrixes = dict()
        self.Jacobi = sm.zeros(len(self.equations), len(self.variables))
        for eq_index, eq in enumerate(self.equations):
            for var_index, var in enumerate(self.variables):
                eq_d = sm.diff(eq, var)
                self.Jacobi[eq_index, var_index] = eq_d

        for var_index, var in enumerate(self.variables):
            jacobi_var = deepcopy(self.Jacobi)
            jacobi_var[:, var_index] = self.equations
            self.jacobi_matrixes[str(var)] = jacobi_var

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
        for var, jacobi_var in self.jacobi_matrixes.items():
            jacobi = deepcopy(self.Jacobi)

            jacobi = jacobi.subs(approximation.items()).det()
            jacobi_var = jacobi_var.subs(approximation.items()).det()

            solve_system[var] = jacobi_var / jacobi

        for var, value in solve_system.items():
            approximation[var] = approximation[var] - value

        return approximation

    def _subs_approximation_structure_graph(self, approximation):
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
        for node in self.gpa.nodes():
            node_info = self.gpa.nodes[node]
            for var, value in approximation.items():
                if str(node_info['P']) == var:
                    node_info['P'] = value

        for edge in self.gpa.edges():
            edge_info = self.gpa.edges[edge]
            for var, value in approximation.items():
                if str(edge_info['x']) == var:
                    edge_info['x'] = value

    def solve_equations(self, start_approximation=None, eps=0.001, iterations=25):
        """

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

        if not start_approximation:
            raise KeyError('Начальное приближение не задано!')

        start_approximation_vars = list(start_approximation.keys())

        for var in self.var_literals:
            if var not in start_approximation_vars:
                raise KeyError('Не задано начальное значение для ' + str(var))

        approximation = start_approximation
        errors = list()

        for _ in range(iterations):
            approximation = self._solve_step(approximation)
            error = deepcopy(self.equations.subs(approximation.items())).norm()
            errors.append(error)

            if error < eps:
                break

        points = -int(math.log10(eps))

        for var in approximation:
            approximation[var] = round(approximation[var], points)

        self.find_approximation = approximation
        self.solving_errors = errors
        self._subs_approximation_structure_graph(self.find_approximation)

        return approximation

    def construct_sense_matrix(self):
        """
        Constructing all sensitivities matrixes
        :return:
        """
        self.p_var = self.P[self.known_q_nodes, :]
        self.p_fix = self.P[self.known_p_nodes, :]
        self.q_var = self.Q[self.known_p_nodes, :]
        self.q_fix = self.Q[self.known_q_nodes, :]

        a_q = self.A[self.known_q_nodes, :]
        a_p = self.A[self.known_p_nodes, :]

        a_f_q = self.AF[self.known_q_nodes, :]
        a_f_p = self.AF[self.known_p_nodes, :]

        a_l_q = self.AL[self.known_q_nodes, :]
        a_l_p = self.AL[self.known_p_nodes, :]

        d_f = self.DF
        d_l = self.DL

        self.full_approximation = deepcopy(self.base_approximation)
        self.full_approximation.update(self.find_approximation)

        self.M = a_q * (d_f * a_f_q.transpose() + d_l * a_l_q.transpose())
        self.inv_M = deepcopy(self.M).inv()
        self.M_PQ = a_p * (d_f * a_f_q.transpose() + d_l * a_l_q.transpose())
        self.M_QP = a_q * (d_f * a_f_p.transpose() + d_l * a_l_p.transpose())
        self.M_PP = a_p * (d_f * a_f_p.transpose() + d_l * a_l_p.transpose())

        self.M = self.M.subs(self.full_approximation.items())
        self.inv_M = self.inv_M.subs(self.full_approximation.items())
        self.M_QP = self.M_QP.subs(self.full_approximation.items())
        self.M_PQ = self.M_PQ.subs(self.full_approximation.items())
        self.M_PP = self.M_PP.subs(self.full_approximation.items())

        self.dp_var = self.inv_M * self.q_fix - self.inv_M * self.M_QP * self.p_fix
        self.dq_var = self.M_PQ * self.inv_M * self.q_fix + (self.M_PP - self.M_PQ * self.inv_M * self.M_QP) * self.p_fix
