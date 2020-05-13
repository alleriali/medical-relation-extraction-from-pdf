import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
class MergeRelation():
    def __init__(self):
        self.kg_df = None
        self.relation_counts = None

    def process_df(self,df):
        print(df.dtypes)
        lemmatizer = WordNetLemmatizer()
        for i, row in df.iterrows():
            source_name = " ".join(
                [lemmatizer.lemmatize(token) for token in word_tokenize(row['source'].lower())])
            target_name = " ".join(
                [lemmatizer.lemmatize(token) for token in word_tokenize(row['target'].lower())])
            if source_name==target_name:
                df.drop(index=i)
            else:
                df.at[i, 'source'] = source_name
                df.at[i, 'target'] = target_name

        return df



    def get_graph(self,kg_df):
        frames = [self.kg_df, kg_df]
        self.kg_df = pd.concat(frames,sort=False)






    def vote_relations(self):

        self.df_agg = self.kg_df.groupby(['source', 'target', 'edge'], as_index=False).agg(
            {'sent': {'counts':'count','unique_counts': 'nunique', 'distinct_sents': lambda x: set(x)},
             'pdf': {'counts':'count','unique_counts': 'nunique', 'distinct_pdfs': lambda x: set(x)}})
        self.voted_by_sents = self.df_agg[self.df_agg.sent.unique_counts > 1]
        self.voted_by_pdfs = self.df_agg[self.df_agg.pdf.unique_counts > 1]


        self.relation_counts = self.kg_df.groupby(['source', 'target', 'edge'], as_index=False).size().reset_index(). \
            rename(columns={0: 'counts'})  ## get the counts for each relation ,output is a dataframe
        idx = self.relation_counts.groupby(['source','target'])['counts'].transform(max) == self.relation_counts['counts'] #get the index of relations of same entities with max counts
        vote_relations = self.relation_counts[idx]
        self.vote_relations = vote_relations.sort_values(by='counts', ascending=False)


    def get_final_graph(self):
        self.G= nx.from_pandas_edgelist(self.vote_relations, "source", "target",edge_attr=True, create_using=nx.DiGraph())

        edges_to_be_removed = set()
        for row in self.G.edges.data():
            u = row[0]
            v = row[1]
            if self.G.has_edge(v, u):
                if self.G[u][v]['edge'] == self.G[v][u]['edge']:

                    if self.G[u][v]['counts'] >= self.G[v][u]['counts']:
                        edges_to_be_removed.add((v, u))
                    else:
                        edges_to_be_removed.add((u, v))

        self.G.remove_edges_from(edges_to_be_removed)

    def show_graph(self):
        self.pos = nx.spring_layout(self.G)
        plt.figure(figsize=(48, 48))
        edge_labels = nx.get_edge_attributes(self.G,'edge')
        nx.draw(self.G,pos=self.pos, with_labels=True, node_color='skyblue',font_size=15,scale=2, k=4)
        nx.draw_networkx_edge_labels(self.G,pos = self.pos,edge_labels=edge_labels,font_color ='green')
        plt.show()




