#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.tasks.preprocessing_funcs import load_dataloaders
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained
import logging
from argparse import ArgumentParser
from src.tasks.visualization import Graph
from src.tasks.pdf_to_txt import PdfToTxt
from src.tasks.merge_relations import MergeRelation
from src.tasks.graph_visualization import GraphVisualization
from src.tasks.get_training_data import get_candidates_for_train
import glob,os
import pandas as pd
'''
This fine-tunes the BERT model on SemEval task 
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default='/home/ying/PycharmProjects/biobert/datasets/RE/three_relations/train.tsv', \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='/home/ying/PycharmProjects/biobert/datasets/RE/three_relations/test.tsv', \
                        help="test data .txt file path")
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=3, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=10, help="No of epochs")
    parser.add_argument("--lr", type=float, default=2*0.00005, help="learning rate")
    parser.add_argument("--model_no", type=int, default=2, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT''')
    
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")
    parser.add_argument("--freeze", type=int, default=0, help='''1: Freeze most layers until classifier layers\
                                                                \n0: Don\'t freeze \
                                                                (Probably best not to freeze if GPU memory is sufficient)''')
    args = parser.parse_args()

    
    if args.train == 1:
        net = train_and_fit(args)
    print("aaaa")
    if args.infer == 1:

        inferer = infer_from_trained(args, detect_entities=True)

        test2 = "exudry ® , omniderm ® , vigilon ® , duoderm ® , mepitel ® ) may aid in healing and reduce pain ."
        pred = inferer.infer_sentence(test2, detect_entities=True)
        print(pred)

        test3 = "Duplication of this publication or parts thereof is permitted only under the provisions of the Copyright Law of the Publisher’s location, in its current version, and permission for use must always be obtained from Springer."
        pred = inferer.infer_sentence(test3, detect_entities=True)
        print(pred)

        test4 = "moreover , uft has proved to be effective for inoperable advanced malignancies such as colorectal cancer, especially in combination with leucovorin or cisplatin."
        pred = inferer.infer_sentence(test4, detect_entities=True)
        print(pred)

        test4 = "cancer can not be treated by dienogest."
        pred = inferer.infer_sentence(test4, detect_entities=True)
        print(pred)

        test5 = "In more severe cases with high fever or marked prostration , hospitalization may be needed with IV acyclovir , antibiotics , ﬂuids , and pain medications ."
        pred = inferer.infer_sentence(test5, detect_entities=True)
        print(pred)

        test6 ="GIANT CELL TUMOR OF THE TENDON SHEATH A giant cell tumor of the tendon sheath is the most common tumor of the hand and presents with a ﬁrm enlarging nodule on the ﬁngers ."
        pred = inferer.infer_sentence(test6, detect_entities=True)
        print(pred)

        test7 = "topical corticosteroids may improve the dermatitis, and chronic administration of oral acyclovir is appropriate for patients with eh "
        pred = inferer.infer_sentence(test7, detect_entities=True)
        print(pred)

        #
        # # # sents_for_pretraining =[]
        merged = MergeRelation()
        for file in glob.glob('data/PDF_FOR_PARSE/*.pdf'):
            csv_file = file.replace('.pdf', '.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
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
                    sents_df = pd.DataFrame(input_sents)
                    sents_df.to_csv(sents_path)
                for sent in input_sents:
                    pred = inferer.infer_sentence(sent,detect_entities=True)
                    graph.add_edges(pred)

                graph.get_df()
                csv_path = file.replace('.pdf', '.csv')
                graph.kg_df.to_csv(csv_path)
                merged.get_df(graph.kg_df)

        merged.kg_df.to_csv('./results/relation_result/DF_full_without_lemm.csv', index=False)
        merged.vote_relations()
        merged.show_graph(merged.kg_df)
        merged.voted_by_pdfs.to_csv('./results/relation_result/DF_vote_by_pdfs.csv',index=False)
        merged.voted_by_sents.to_csv('./results/relation_result/DF_vote_by_sentences.csv',index=False)
        merged.show_voted_by_pdf()
        merged.show_voted_by_sents()
        GraphVisual = GraphVisualization(merged.G)

        while True:
            request = input("please input relation or entity or infer or exit:\n")
            if request.lower() == 'relation':
                relation_name = input(
                    "Type one realtion type (treat or side_effect or contraindication) you are interested in:\n")
                GraphVisual.show_gragh_by_edge(relation_name)
            elif request.lower() == 'entity':
                node = input(
                    "Type the node or nodes you are interested in, input nodes with comma to separate them:\n")
                if node.find(',') != -1:
                    node = node.split(',')
                    node = list(node)
                    GraphVisual.node_adj_and_shown(node)
                else:
                    GraphVisual.node_adj_and_shown(node)
            elif request.lower() == 'infer':
                sent = input("input a sentence for infer:\n")
                print(inferer.infer_sentence(test4, detect_entities=True))
            else:
                break

