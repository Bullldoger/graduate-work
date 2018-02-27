import matplotlib.pyplot as plt
import networkx as nx


def disp_error_plot(fig_size=10, errors=[]):
    """

    :param errors: list of errors
    :param fig_size: plot size
    :return:
    """
    plt.figure(figsize=(fig_size, fig_size))

    plt.plot(range(len(errors)), errors)
    plt.show()


def disp_graph_with_custom_labels(gr=None, nodes_labels={}, edges_labels={},
                                  fig_size=8, node_label=None, edge_label=None):
    """

    :param gr: networkx graph
    :type gr: nx.Graph() and all child classes
    :param nodes_labels: dict with labels
    :param edges_labels: dict with labels
    :param fig_size: plot size
    :param node_label: node attribute label
    :param edge_label: edge attribute label
    :return:
    """
    plt.figure(figsize=(fig_size, fig_size))

    if not edge_label == None and not edge_label == '':
        edges_labels = nx.get_edge_attributes(G=gr, name=edge_label)

    if not node_label == None and not node_label == '':
        nodes_labels = nx.get_node_attributes(G=gr, name=node_label)

    layout = nx.shell_layout(gr)

    nx.draw(G=gr, pos=layout, node_size=1500)
    nx.draw_networkx_edge_labels(G=gr,
                                 pos=layout,
                                 edge_labels=edges_labels,
                                 font_family='sans-serif')
    nx.draw_networkx_labels(G=gr,
                            pos=layout,
                            labels=nodes_labels,
                            font_family='sans-serif')

    plt.show()