import networkx as nx
import pandas as pd
import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_agraph import graphviz_layout
import re

class Graph():

    def __init__(self,pdf_name):
        self.sources = []
        self.targets = []
        self.edges = []
        self.sents=[]
        self.pdf = pdf_name

    def add_edge(self, relations):

        for relation in relations:
            tag = relation[0]
            index = re.search('\(', tag).start()
            realtion_name = tag[:index]

            if len(re.findall('\(e2,e1\)',tag)) != 0:
                source = relation[1]['e2']
                target = relation[1]['e1']
            else:
                source = relation[1]['e1']
                target = relation[1]['e2']
            sent = relation[1]['sent']

            self.sources.append(source)
            self.targets.append(target)
            self.edges.append(realtion_name)
            self.sents.append(sent)
    def get_df(self):
        self.kg_df = pd.DataFrame({'source': self.sources, 'target': self.targets, 'edge': self.edges,'sent':self.sents,'pdf':[self.pdf]*len(self.edges)})


    def show_graph(self):
        self.G = nx.from_pandas_edgelist(self.kg_df, "source", "target",
                                    "edge", create_using=nx.DiGraph())
        self.pos = nx.spring_layout(self.G)
        plt.figure(figsize=(48, 48))
        edge_labels = nx.get_edge_attributes(self.G,'edge')
        nx.draw(self.G,pos=self.pos, with_labels=True, node_color='skyblue',font_size=15,scale=2, k=4)
        nx.draw_networkx_edge_labels(self.G,pos = self.pos,edge_labels=edge_labels,font_color ='green')
        plt.show()

    def node_adj_and_shown(self, node):
        nodes = set()

        if isinstance(node, list):
            #print("the adjacent nodes to input: " + str(self.G.edges(node)))
            for n in node:
                if n in self.G:
                    print("the adjacent nodes to the node " + "'"+ n +"'" +":" +str(list(self.G[n])))
                    nodes.add(n)
                    nodes.update([i for i in self.G[n]])
                else:
                    print("there are no adjacent nodes to the node " + "'"+ n+"'" )
                    continue
        else:
            if node in self.G:
                print("the adjacent nodes to the node " + "'"+node+"'" +":" + str(list(self.G[node])))
                nodes.add(node)
                nodes.update([i for i in self.G[node]])
            else:
                print("there are no adjacent nodes to the node "+ "'"+node+"'" )
        print(nodes)
        if len(nodes)>=2:
            subgraph = self.G.subgraph(nodes)
            edge_labels = nx.get_edge_attributes(subgraph, 'edge')
            plt.figure()
            nx.draw_networkx(subgraph, pos=self.pos,with_labels=True)
            nx.draw_networkx_edge_labels(subgraph, pos=self.pos,edge_labels=edge_labels)
            plt.show()


    def show_gragh_by_edge(self,relation_name):
        graph = nx.from_pandas_edgelist(self.kg_df[self.kg_df['edge'] == relation_name], "source", "target",
                                    "edge", create_using=nx.DiGraph())

        plt.figure(figsize=(24, 24))
        pos = nx.spring_layout(graph, k=2)  # k regulates the distance between nodes
        edge_labels = nx.get_edge_attributes(graph, 'edge')
        nx.draw(graph, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=self.pos)
        nx.draw_networkx_edge_labels(graph, pos=self.pos, edge_labels=edge_labels, font_color='green')
        plt.show()

