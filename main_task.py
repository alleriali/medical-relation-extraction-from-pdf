#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""

from src.tasks.preprocessing_funcs import load_dataloaders
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained
import logging
from argparse import ArgumentParser
from src.tasks.visualization import Graph
from src.tasks.pdf_to_txt import PdfToTxt
from src.tasks.merge_relations import MergeRelation
from src.tasks.graph_visualization import GraphVisualization
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
    parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/treat_cause_p_train.txt', \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/treat_cause_p_test.txt', \
                        help="test data .txt file path")
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=8, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=50, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--model_no", type=int, default=2, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT''')
    
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")
    parser.add_argument("--freeze", type=int, default=1, help='''1: Freeze most layers until classifier layers\
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
        print("222")
        print(pred)

        test3 = "Duplication of this publication or parts thereof is permitted only under the provisions of the Copyright Law of the Publisher’s location, in its current version, and permission for use must always be obtained from Springer."
        pred = inferer.infer_sentence(test3, detect_entities=True)
        print("333")
        print(pred)

        print("444")
        test4 = "moreover , uft has proved to be effective for inoperable advanced malignancies such as colorectal cancer, especially in combination with leucovorin or cisplatin."
        pred = inferer.infer_sentence(test4, detect_entities=True)
        print(pred)
        print("444")
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
        # # sents_for_pretraining =[]
        merged = MergeRelation()
        for file in glob.glob('./data/Test_PDF/*'):
            if file.find('.pdf') != -1:
                print(file)
                csv_file = file.replace('.pdf', '.csv')
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    merged.get_graph(df)
                else:
                    file_name = os.path.basename(file)
                    graph = Graph(file_name)
                    Pdf2txt = PdfToTxt(file)
                    input_sents = Pdf2txt.get_processed_sents()
                # sents_for_pretraining.extend(input_sents)
                    for sent in input_sents:
                        sent = sent.lower()
                        pred = inferer.infer_sentence(sent,detect_entities=True)
                        if pred is not None:
                            graph.add_edge(pred)
                    graph.get_df()
                    csv_path = file.replace('.pdf', '.csv')
                    graph.kg_df.to_csv(csv_path)
                    merged.get_graph(graph.kg_df)
        # for file in glob.glob('./data/PDF_For_Train/*'):
        #     if file.find('.csv')!=-1:
        #         kg_df = pd.read_csv(file)
        #         # merged.get_graph(kg_df)
        # # #merged.kg_df = merged.process_df(merged.kg_df)
        # # # # with open('./data/sents_for_pretraining.txt','w+') as f:
        # # # #     for sent in sents_for_pretraining:
        # # # #         f.write(sent+'\n'+'\n')
        # # #merged.kg_df = merged.process_df(merged.kg_df)
        # # # merged.kg_df.to_csv('./data/PDF_For_Train/DF_full_without_lemm.csv', index=False)
        merged.vote_relations()
        # # # # # # merged.get_final_graph()
        # # # # # # merged.show_graph()
        # # # # #
        merged.voted_by_pdfs.to_csv('./data/Test_PDF/DF_vote_by_pdfs.csv',index=False)
        merged.voted_by_sents.to_csv('./data/Test_PDF/DF_vote_by_sentences.csv', index=False)
        # # GraphVisua = GraphVisualization(merged.G)

        # print('55555')
        # while True:
        #     # sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
        #     # if sent.lower() in ['quit', 'exit']:
        #     #     break
        #     # inferer.infer_sentence(sent, detect_entities=True)
        #     relation_name = input("Type the the name of the realtion you are interested in('quit' or 'exit' to terminate):\n")
        #     if relation_name.lower() not in ['quit', 'exit']:
        #         GraphVisua.show_gragh_by_edge(relation_name)
        #     else:
        #         break
        #     node = input("Type the node or nodes you are interested in, input nodes with comma to separate them('quit' or 'exit' to terminate):\n")
        #     if node.find(',') != -1:
        #         node = node.split(',')
        #         node = list(node)
        #         GraphVisua.node_adj_and_shown(node)
        #     elif node.lower() in ['quit', 'exit']:
        #         break
        #     else:
        #         GraphVisua.node_adj_and_shown(node)
