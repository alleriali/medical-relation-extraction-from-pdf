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



    def get_df(self,kg_df):
        frames = [self.kg_df, kg_df]
        self.kg_df = pd.concat(frames,sort=False)






    def vote_relations(self):
        def get_distinct(x):
            return set(x)

        self.G = nx.from_pandas_edgelist(self.kg_df, "source", "target", edge_attr=True,
                                         create_using=nx.DiGraph())
        self.df_agg = self.kg_df.groupby(['source', 'target', 'relation'], as_index=False).agg(
            {'sent': [('s_counts','count'),('s_unique_counts', 'nunique'), ('distinct_sents', lambda x: set(x))],
             'pdf': [('p_counts','count'),('p_unique_counts', 'nunique'), ('distinct_pdfs',lambda x: set(x))]})
        # self.df_agg = self.kg_df.groupby(['source', 'target', 'relation'], as_index=False).agg(
        #     {'sent': {'s_counts':'count','s_unique_counts': 'nunique', 'distinct_sents': lambda x: set(x)},
        #      'pdf': {'p_counts':'count','p_unique_counts': 'nunique', 'distinct_pdfs': lambda x: set(x)}})
        self.voted_by_sents = self.df_agg[self.df_agg.sent.s_unique_counts > 1]
        self.voted_by_pdfs = self.df_agg[self.df_agg.pdf.p_unique_counts > 1]


        self.relation_counts = self.kg_df.groupby(['source', 'target', 'relation'], as_index=False).size().reset_index(). \
            rename(columns={0: 'counts'})  ## get the counts for each relation ,output is a dataframe
        idx = self.relation_counts.groupby(['source','target'])['counts'].transform(max) == self.relation_counts['counts'] #get the index of relations of same entities with max counts

        vote_relations = self.relation_counts[idx]
        un_vote_relations = self.relation_counts[~idx]
        self.vote_relations = vote_relations.sort_values(by='counts', ascending=False)
        self.un_vote_relations = un_vote_relations.sort_values(by='counts', ascending=False)



    def get_final_graph(self):
        self.G= nx.from_pandas_edgelist(self.voted_by_pdfs, "source", "target",edge_attr=True, create_using=nx.DiGraph())

        edges_to_be_removed = set()
        for row in self.G.edges.data():
            u = row[0]
            v = row[1]
            if self.G.has_edge(v, u):
                if self.G[u][v]['relation'] == self.G[v][u]['relation']:

                    if self.G[u][v]['counts'] >= self.G[v][u]['counts']:
                        edges_to_be_removed.add((v, u))
                    else:
                        edges_to_be_removed.add((u, v))

        self.G.remove_edges_from(edges_to_be_removed)

    def show_graph(self,KG_df,title='Extracted Relations'):
        KG = nx.from_pandas_edgelist(KG_df, "source", "target",edge_attr=True, create_using=nx.DiGraph())
        print(title,KG.number_of_edges())
        pos = nx.spring_layout(KG)
        plt.figure(figsize=(16, 16))
        plt.title(title,fontsize=20)
        edge_labels = nx.get_edge_attributes(KG,'relation')
        print(edge_labels)
        nx.draw(KG,pos=pos, with_labels=True, node_size=400, node_color='skyblue',font_size=17,scale=1, k=5)
        nx.draw_networkx_edge_labels(KG,pos = pos,edge_labels=edge_labels,font_color ='green')
        plt.show()

    def show_voted_by_pdf(self):
        title = "Relations extracted from different PDFs"
        voted_by_pdfs = self.voted_by_pdfs.set_index(['source','target','relation'])
        voted_by_pdfs.columns = voted_by_pdfs.columns.droplevel()
        voted_by_pdfs = voted_by_pdfs.reset_index()
        voted_by_pdfs = voted_by_pdfs.drop(['s_counts','s_unique_counts','distinct_sents','p_counts','distinct_pdfs'], axis=1)
        voted_by_pdfs = voted_by_pdfs.rename({'p_unique_counts': 'PDFs'}, axis=1)
        self.show_graph(voted_by_pdfs,title)

    def show_voted_by_sents(self):
        title = "Relations extracted from different sentences"
        voted_by_sents = self.voted_by_sents.set_index(['source','target','relation'])
        voted_by_sents.columns = voted_by_sents.columns.droplevel()
        voted_by_sents = voted_by_sents.reset_index()
        voted_by_sents = voted_by_sents.drop(['s_counts','distinct_sents','p_counts','p_unique_counts','distinct_pdfs'], axis=1)
        voted_by_sents = voted_by_sents.rename({'s_unique_counts': 'sents'}, axis=1)
        self.show_graph(voted_by_sents,title)







