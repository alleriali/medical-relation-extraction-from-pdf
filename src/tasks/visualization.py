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
        self.relations = []
        self.sents=[]
        self.pdf = pdf_name

    def add_edges(self,sents_with_relations:list):
        for sent_with_relation in sents_with_relations:
            self.sents.append(sent_with_relation['sent'])
            self.sources.append(sent_with_relation['chemical'])
            self.targets.append(sent_with_relation['disease'])
            self.relations.append(sent_with_relation['relation'])

    def get_df(self):
        self.kg_df = pd.DataFrame({'source': self.sources, 'target': self.targets, 'relation': self.relations,'sent':self.sents,'pdf':[self.pdf]*len(self.sents)})


    def show_graph(self):
        self.G = nx.from_pandas_edgelist(self.kg_df, "source", "target",
                                    "relation", create_using=nx.DiGraph())
        self.pos = nx.spring_layout(self.G)
        plt.figure(figsize=(48, 48))
        edge_labels = nx.get_edge_attributes(self.G,'relation')
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
            edge_labels = nx.get_edge_attributes(subgraph, 'relation')
            plt.figure()
            nx.draw_networkx(subgraph, pos=self.pos,with_labels=True)
            nx.draw_networkx_edge_labels(subgraph, pos=self.pos,edge_labels=edge_labels)
            plt.show()


    def show_gragh_by_edge(self,relation_name):
        graph = nx.from_pandas_edgelist(self.kg_df[self.kg_df['relation'] == relation_name], "source", "target",
                                    "relation", create_using=nx.DiGraph())

        plt.figure(figsize=(24, 24))
        pos = nx.spring_layout(graph, k=2)  # k regulates the distance between nodes
        edge_labels = nx.get_edge_attributes(graph, 'relation')
        nx.draw(graph, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=self.pos)
        nx.draw_networkx_edge_labels(graph, pos=self.pos, edge_labels=edge_labels, font_color='green')
        plt.show()

