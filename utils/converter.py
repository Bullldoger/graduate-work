import numpy
import sympy as sm

import json


def read_graph(path='', num=-1):
    with open(path, 'r') as input:
        edges = dict()
        line = input.readline()

        gpa_n = {
            'edges': {
                'edges_info': {
                }
            },
            'nodes': {
                'nodes_info': {
                }
            }
        }

        params = list(map(lambda x: int(x),
                          line.split(' ')))
        V, E = params[0], params[1]

        internal_nodes = numpy.zeros([2, V])

        gpa_n['nodes']['count'] = V
        gpa_n['edges']['count'] = E

        for i in range(V):

            params = list(map(lambda x: int(x),
                              input.readline().split(' ')))
            node = params[0]

            node_info = {
                'P': {
                    'P': None,
                    'P_max': 100,
                    'P_min': 0
                },
                'Q': {
                    'Q': None,
                    'Q_max': 100,
                    'Q_min': 0
                },
                'name': ''
            }

            if len(params) == 1:
                p_i = sm.symbols('p' + str(i + 1))
                known = False
            else:
                p_i = params[1]
                node_info['P']['P'] = p_i
                known = True

            node_info['name'] = str(i + 1)

            gpa_n['nodes']['nodes_info'][str(node)] = node_info

        for i in range(E):
            params = list(map(lambda x: int(x),
                              input.readline().split(' ')))

            u = int(params[0])
            v = int(params[1])
            l = params[2]
            d = params[3]

            edge_info = {
                "A": None,
                "u": None,
                "v": None,
                "x": None
            }

            internal_nodes[0, u - 1] += 1
            internal_nodes[1, v - 1] += 1

            u = str(u)
            v = str(v)

            if u in edges:
                edges[u].append(str(i + 1))
            else:
                edges[u] = [str(i + 1)]

            if v in edges:
                edges[v].append(str(i + 1))
            else:
                edges[v] = [str(i + 1)]

            edge_info['u'] = str(u)
            edge_info['v'] = str(v)
            edge_info['A'] = l * d

            if len(params) == 5:
                edge_info['x'] = params[4]

            gpa_n['edges']['edges_info'][str(i + 1)] = edge_info

        internal_nodes = numpy.logical_and(internal_nodes[0, :], internal_nodes[1, :])

        for i, is_internal in enumerate(internal_nodes):
            node = str(i + 1)
            if not is_internal:
                gpa_n['nodes']['nodes_info'][node]['Q']['Q'] = gpa_n['edges']['edges_info'][edges[node][0]]['x']
            else:
                gpa_n['nodes']['nodes_info'][node]['Q']['Q'] = 0

        f_name = 'gpa-' + str(num)
        path = 'models/' + f_name + '.json'

        with open(path, 'w') as fp:
            json.dump(gpa_n, fp)
