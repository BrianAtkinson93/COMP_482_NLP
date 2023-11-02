from dawg_class import DAWGNode
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional, Union


def build_simple_dawg(word_list: List[str]) -> DAWGNode:
    """
    Builds a simplified DAWG from a list of words.

    :param word_list: List of words to include in the DAWG.
    :return: The root node of the constructed DAWG.
    """
    root = DAWGNode()

    for word in word_list:
        current_node = root
        for char in word:
            next_node = current_node.get_edge(char)
            if next_node is None:
                next_node = DAWGNode()
                current_node.add_edge(char, next_node)
            current_node = next_node
        current_node.is_terminal = True  # Mark the end of a word

    return root


def dawg_lookup(root: DAWGNode, word: str) -> bool:
    """
    Looks up a word in the DAWG starting from the root node.

    :param root: The root node of the DAWG.
    :param word: The word to look up.
    :return: True if the word exists in the DAWG, False otherwise.
    """
    current_node = root

    for char in word:
        next_node = current_node.get_edge(char)
        if next_node is None:
            return False
        current_node = next_node

    return current_node.is_terminal


def visualize_dawg(node: DAWGNode, graph: Optional[nx.DiGraph] = None, parent: Optional[Union[int, str]] = None,
                   edge_label: Optional[str] = None) -> nx.DiGraph:
    """
    Visualizes the DAWG starting from a node.

    :param node: The current node being visited.
    :param graph: The DiGraph object being constructed (used for recursion).
    :param parent: The parent node id (used for recursion).
    :param edge_label: The label for the edge from parent to node (used for recursion).
    :return: The completed DiGraph object.
    """
    if graph is None:
        graph = nx.DiGraph()

    node_id = id(node)  # Unique identifier for each node

    graph.add_node(node_id)
    if parent is not None:
        graph.add_edge(parent, node_id, label=edge_label)

    for char, next_node in node.edges.items():
        visualize_dawg(next_node, graph, node_id, edge_label=char)

    return graph


def main():
    """
    Main function to build, visualize, and test the DAWG.
    """
    words = ["bat", "batman", "batwoman", "man", "woman"]
    root = build_simple_dawg(words)

    # Visualize the DAWG
    plt.figure(figsize=(20, 20))  # Increase figure size
    graph = visualize_dawg(root)

    pos = nx.spring_layout(graph, seed=42)  # Added seed for reproducibility
    nx.draw(graph, pos, with_labels=True, node_size=1000, node_color="red", font_size=16)

    labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=16)

    plt.show()

    # Test the DAWG
    print(dawg_lookup(root, "bat"))  # Should print True
    print(dawg_lookup(root, "batman"))  # Should print True
    print(dawg_lookup(root, "batwoman"))  # Should print True
    print(dawg_lookup(root, "man"))  # Should print True
    print(dawg_lookup(root, "woman"))  # Should print True
    print(dawg_lookup(root, "batwomanx"))  # Should print False


if __name__ == "__main__":
    main()
