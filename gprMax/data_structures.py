import copy

class Node:

    def __init__(self, name='noname'):
        self.children = []
        self.name = name
        self.marked = False

    def add(self, *args):
        for node in args:
            self.children.append(node)

class TreeWalker:

    def getBreadthFirstNodes(self, tree):

        visited = copy.copy(tree.children)
        q = copy.copy(tree.children)

        while q:
            v = q.pop(0)

            for w in v.children:

                if not w.marked:
                    w.marked = True
                    q.append(w)
                    visited.append(w)
        return visited


    def getDepthFirstNodes(self, tree):
        nodes = []
        for child in tree.children:
            nodes.append(child)
            nodes += self.getDepthFirstNodes(child)
        return nodes


if __name__ == '__main__':

    alpha = list(map(chr, range(97, 107)))
    nodes = {c: Node(c) for c in alpha}

    nodes['a'].children.append(nodes['b'])
    nodes['a'].children.append(nodes['c'])

    nodes['b'].children.append(nodes['d'])
    nodes['b'].children.append(nodes['e'])
    nodes['b'].children.append(nodes['f'])

    nodes['c'].children.append(nodes['g'])

    nodes['e'].children.append(nodes['h'])
    nodes['e'].children.append(nodes['i'])

    nodes['g'].children.append(nodes['j'])

    tw = TreeWalker()
    nl = tw.getBreadthFirstNodes(nodes['a'])
    print([i.name for i in nl])

    nl2 = tw.getDepthFirstNodes(nodes['a'])
    print([i.name for i in nl2])
