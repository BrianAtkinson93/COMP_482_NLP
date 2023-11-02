class DAWGNode:
    def __init__(self):
        self.edges = {}  # Dictionary to hold edges where key=character, value=DAWGNode
        self.is_terminal = False  # Indicates whether this node is the end of a word

    def add_edge(self, char, node):
        self.edges[char] = node

    def get_edge(self, char):
        return self.edges.get(char, None)
