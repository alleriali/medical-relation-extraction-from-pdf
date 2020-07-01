# # sents_for_pretraining =[]
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained
import logging
from argparse import ArgumentParser
from src.tasks.visualization import Graph
from src.tasks.pdf_to_txt import PdfToTxt
from src.tasks.merge_relations import MergeRelation
from src.tasks.graph_visualization import GraphVisualization
from src.tasks.get_training_data import get_candidates_for_train
from nltk import word_tokenize
from src.BioBERT_NER_RE import ner_lib,re_lib
import glob,os,re,time
import pandas as pd


def get_abbreviations_lookup():
    abbreviations = dict()
    abbreviations_df = pd.read_csv('./data/medical_abbreviations/Common_Abbreviations.tsv',sep='\t')
    for idx, row in abbreviations_df.iterrows():
        key = row['Abbreviation']
        abbreviations[key] = row['Stands for']
    return abbreviations

def get_entities_relations(sents):
    ner_time_start = time.time()
    sents_with_entities = ner_lib.get_annotated_sents_with_entities(sents)
    print("the time for running NER:",time.time()-ner_time_start)
    sents_for_test =[]
    print(sents_with_entities)
    for sent_with_entity in sents_with_entities:
        annotated_sent = sent_with_entity['sent']
        print(annotated_sent)
        print(type(annotated_sent))
        sent = re.sub('\[D\].*\[/D\]', 'DISEASE', annotated_sent)
        sent_for_test = re.sub('\[C\].*\[/C\]', 'CHEMICAL', sent)
        sents_for_test.append(sent_for_test)
    re_time_start = time.time()
    predicted_relations = re_lib.get_predicted_relation(sents_for_test)
    print("the time for running RE:", time.time() - re_time_start)
    for i,sent_with_entity in enumerate(sents_with_entities):
        sent_with_entity['relation'] = predicted_relations[i]
    return sents_with_entities

if __name__ == "__main__":
    #sents = ['moreover , uft has proved to be effective for inoperable advanced malignancies such as colorectal cancer, especially in combination with leucovorin or cisplatin.', \
             #"topical corticosteroids may improve the dermatitis, and chronic administration of oral acyclovir is appropriate for patients with eh "]
    # sents =['Drugs known to cause lichen planus - like ( lichenoid ) eruptions include gold ( sodium aurothiomalate ) , penicillamine , and mepacrine .']
    # get_entities_relations(sents)
    abbreviations_lookup = get_abbreviations_lookup()


    def replace_abbreviations(sents):
        new_sents =[]
        for sent in sents:
            new_tokens = []
            tokens = word_tokenize(sent)
            for token in tokens:
                if token in abbreviations_lookup:
                    print(token)
                    new_tokens.append(abbreviations_lookup[token].lower())
                else:
                    new_tokens.append(token)
            sent = " ".join(new_tokens)
            new_sents.append(sent)
        return new_sents


    merged = MergeRelation()
    for file in glob.glob('data/PDF_FOR_TEST_40/*.pdf'):
        csv_file = file.replace('.pdf', '.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(csv_file,len(df))
            merged.get_df(df)
        else:
            file_name = os.path.basename(file)
            sents_path = file.replace('.pdf', '_sents.csv')
            print(sents_path)
            graph = Graph(file_name)
            input_sents = []
            if os.path.exists(sents_path):
                df_sents = pd.read_csv(sents_path,index_col=0)
                for idx, row in df_sents.iterrows():
                    input_sents.append(row[0])
            else:
                Pdf2txt = PdfToTxt(file,is_pdf=True)
                input_sents = Pdf2txt.get_processed_sents()
                input_sents = replace_abbreviations(input_sents)
                sents_df = pd.DataFrame(input_sents)
                sents_df.to_csv(sents_path)
            graph.add_edges(get_entities_relations(input_sents))
            graph.get_df()
            csv_path = file.replace('.pdf', '.csv')
            graph.kg_df.to_csv(csv_path)
            merged.get_df(graph.kg_df)

    print("merged.kg_df",len(merged.kg_df))
    merged.vote_relations()
    merged.show_graph(merged.kg_df)
    merged.kg_df.to_csv('./results/result_new/DF_full_without_lemm.csv',index=False)
    merged.voted_by_pdfs.to_csv('./results/result_new/DF_vote_by_pdfs.csv',index=False)
    merged.voted_by_sents.to_csv('./results/result_new/DF_vote_by_sentences.csv',index=False)
    merged.vote_relations.to_csv('./results/result_new/vote_relations.csv',index=False)
    merged.un_vote_relations.to_csv('./results/result_new/un_vote_relations.csv', index=False)
    merged.show_voted_by_pdf()
    merged.show_voted_by_sents()
    GraphVisua = GraphVisualization(merged.G)

    while True:

        relation_name = input("Type the realtion type (treat,side_effect,contraindication) you are interested in ('quit' or 'exit' to terminate):\n")
        if relation_name.lower() not in ['quit', 'exit']:
            if relation_name.lower() not in ['treat','side_effect','contraindication']:
                relation_name = input(
                    "Type one realtion type (treat or side_effect or contraindication) you are interested in:\n")
                GraphVisua.show_gragh_by_edge(relation_name)
            else:
                GraphVisua.show_gragh_by_edge(relation_name)
        else:
            break
        node = input("Type the node or nodes you are interested in, input nodes with comma to separate them('quit' or 'exit' to terminate):\n")
        if node.find(',') != -1:
            node = node.split(',')
            node = list(node)
            GraphVisua.node_adj_and_shown(node)
        elif node.lower() in ['quit', 'exit']:
            break
        else:
            GraphVisua.node_adj_and_shown(node)
