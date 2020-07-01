from src.tasks.visualization import Graph
from src.tasks.pdf_to_txt import PdfToTxt
from src.tasks.merge_relations import MergeRelation
from src.tasks.graph_visualization import GraphVisualization
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

def get_entities_relations(sents,infer=False):
    ner_time_start = time.time()
    if infer==True:
        sents_with_entities = ner_lib.infer_on_sentence(sents)
    else:
        sents_with_entities = ner_lib.get_annotated_sents_with_entities(sents)
    if len(sents_with_entities)==2:
        return sents_with_entities[1]
    print("the time for running NER:",time.time()-ner_time_start)
    sents_for_test =[]
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

if __name__ == "__main__":

    abbreviations_lookup = get_abbreviations_lookup()
    merged = MergeRelation()
    for file in glob.glob('data/PDF_FOR_PARSE/*.pdf'):
        csv_file = file.replace('.pdf', '.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(csv_file,len(df))
            merged.get_df(df)
        else:
            file_name = os.path.basename(file)
            sents_path = file.replace('.pdf', '_sents.csv')
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
    merged.kg_df.to_csv('./results/relation_result/DF_full_without_lemm.csv',index=False)
    merged.voted_by_pdfs.to_csv('./results/relation_result/DF_vote_by_pdfs.csv',index=False)
    merged.voted_by_sents.to_csv('./results/relation_result/DF_vote_by_sentences.csv',index=False)
    merged.vote_relations.to_csv('./results/relation_result/vote_relations.csv',index=False)
    merged.un_vote_relations.to_csv('./results/relation_result/un_vote_relations.csv', index=False)
    merged.show_voted_by_pdf()
    merged.show_voted_by_sents()
    GraphVisual = GraphVisualization(merged.G)

    while True:
        request = input("please input relation or entity or infer or exit:\n")
        if request.lower()=='relation':
            relation_name = input(
                "Type one realtion type (treat or side_effect or contraindication) you are interested in:\n")
            GraphVisual.show_gragh_by_edge(relation_name)
        elif request.lower()=='entity':
            node = input(
                "Type the node or nodes you are interested in, input nodes with comma to separate them:\n")
            if node.find(',') != -1:
                node = node.split(',')
                node = list(node)
                GraphVisual.node_adj_and_shown(node)
            else:
                GraphVisual.node_adj_and_shown(node)
        elif request.lower()=='infer':
            sent = input("input a sentence for infer:\n")
            print(get_entities_relations(sent,infer=True))
        else:
            break

