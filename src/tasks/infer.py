#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:44:17 2019

@author: weetee
"""

import pickle
import os
import pandas as pd
import torch
import spacy
from scispacy.abbreviation import AbbreviationDetector
import re
from itertools import permutations,combinations
from src.BioBERT_NER_RE import ner_lib
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class infer_from_trained(object):
    def __init__(self, args=None, detect_entities=False):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        self.detect_entities = detect_entities
        
        if self.detect_entities:
            self.nlp = spacy.load("en_core_sci_md")
            abbreviation_pipe = AbbreviationDetector(self.nlp)
            self.nlp.add_pipe(abbreviation_pipe)
            self.ner = spacy.load("en_ner_bc5cdr_md")
            self.nlp_norm = spacy.load("en_core_web_sm")
        else:
            self.nlp = None
        self.entities_of_interest = ["DISEASE","CHEMICAL"]
        logger.info("Loading tokenizer and model...")
        from .train_funcs import load_state
        
        if args.model_no == 0:
            from ..model.BERT.modeling_bert import BertModel as Model
            model = 'bert-base-uncased'
            lower_case = True
            model_name = 'BERT'
        elif args.model_no == 1:
            from ..model.ALBERT.modeling_albert import AlbertModel as Model
            model = 'albert-base-v2'
            lower_case = False
            model_name = 'ALBERT'
        elif args.model_no == 2:
            from ..model.BIOBERT.modeling_biobert import BiobertModel as Model
            model = 'biobert'
            lower_case = False
            model_name = 'BIOBERT'
        
        self.net = Model.from_pretrained(model, force_download=False, \
                                         task='classification', n_classes_=args.num_classes)
        self.tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        self.net.resize_token_embeddings(len(self.tokenizer))
        if self.cuda:
            self.net.cuda()
        start_epoch, best_pred, amp_checkpoint = load_state(self.net, None, None, self.args, load_best=False)
        logger.info("Done!")
        
        # self.d_id_s = self.tokenizer.convert_tokens_to_ids('[D]')
        # self.d_id_e = self.tokenizer.convert_tokens_to_ids('[/D]')
        # self.c_id_s = self.tokenizer.convert_tokens_to_ids('[C]')
        # self.d_id_s = self.tokenizer.convert_tokens_to_ids('[/C]')
        self.D_id = self.tokenizer.convert_tokens_to_ids('DISEASE')
        self.C_id = self.tokenizer.convert_tokens_to_ids('CHEMICAL')
        self.pad_id = self.tokenizer.pad_token_id
        self.rm = load_pickle("relations.pkl")

        
    def get_all_ent_pairs(self, sent):
        sent_doc = self.nlp(sent)
        sent_ner = self.ner(sent)
        sent_norm = self.nlp_norm(sent)
        organs = ["skin", "kidneys", "heart", "lungs", "pancreas", "gall bladder", "small intestine", "large intestine",
                  "brain", "eyes", "spleen", "tongue", "teeth", "bones", "muscles", "blood", "tympanic membranes",
                  "cochleae", "blood vessels", "bladder", "testes", "ovaries", "cervix", "uterus", "penis", "vagina",
                  "stomach", "esophagus", "trachea", "bronchi", "lymph nodes", "liver", "nerves", "spinal cord",
                  "fingers", "nails", "heads", "hands", "legs"]
        ent_text = set()
        diseases = set()
        chemicals = set()
        self.diseases =set()
        for ent in sent_ner.ents:
            ent_valid = True
            for token in ent:
                if token.pos_!='PROPN' and token.pos_!='NOUN' and token.pos_!='DET':
                    ent_valid = False
                    break
            if not ent_valid:
                continue
            if ent.text.lower() in organs:
                continue
            if ent in sent_norm.ents:
                continue
            for ent_in_doc in sent_doc.ents:
                if ent_in_doc.text in ent.text or ent.text in ent_in_doc.text:
                    if ent.text not in ent_text:
                        ent_text.add(ent.text)
                        if ent.label_ == "DISEASE":
                            diseases.add(ent)
                            self.diseases.add(ent.text)
                        else:
                            chemicals.add(ent)
                        break

        print("dieseas:",diseases)
        print("chemicals:",chemicals)

        pairs = set()
        #only consider pair (diesease,chemical)
        if len(diseases) >= 1 and len(chemicals)>=1:
            for d in diseases:
                for c in chemicals:
                    if d.start<c.start:
                        pairs.add((d,c,'disease','chemical'))
                    else:
                        pairs.add((c,d,'chemical','disease'))

        new_pairs = []
        for pair in pairs:
            if len(pair[0]) > 1:
                p0 = [p for p in pair[0]]
            else:
                p0 = pair[0]
            if len(pair[1]) > 1:
                p1 = [p for p in pair[1]]
            else:
                p1 = pair[1]
            new_pairs.append((p0, p1,pair[2],pair[3]))

        return new_pairs


    def  annotate_sent(self, sent_nlp, e1, e2,e1_type,e2_type):
        annotated = ''
        e1start, e1end, e2start, e2end = 0, 0, 0, 0
        e1start_idx, e2start_idx = 0,0
        if e1_type=='disease':
            e1_s_tag = '[D]'
            e1_e_tag = '[/D]'
        else:
            e1_s_tag = '[C]'
            e1_e_tag = '[/C]'
        if e2_type=='disease':
            e2_s_tag = '[D]'
            e2_e_tag = '[/D]'
        else:
            e2_s_tag = '[C]'
            e2_e_tag = '[/C]'

        for token in sent_nlp:
            if not isinstance(e1, list):
                if (token.text == e1.text) and (e1start == 0) and (e1end == 0):
                    annotated += e1_s_tag + token.text + e1_e_tag
                    e1start, e1end = 1, 1
                    e1start_idx = token.i
                    continue
            else:
                if (token.text == e1[0].text) and (e1start == 0):
                    annotated += e1_s_tag + token.text + ' '
                    e1start += 1
                    e1start_idx = token.i
                    continue
                elif (e1start == 1) and (token.text == e1[-1].text) and (token.i>e1start_idx) and (e1end == 0):
                    print("token i:",token.i)
                    annotated += token.text + e1_e_tag
                    e1end += 1
                    continue
           
            if not isinstance(e2, list):
                if (token.text == e2.text) and (e2start == 0) and (e2end == 0):
                    annotated += e2_s_tag + token.text + e2_e_tag
                    e2start, e2end = 1, 1
                    e2start_idx = token.i
                    continue
            else:
                if (token.text == e2[0].text) and (e2start == 0):
                    annotated += e2_s_tag + token.text + ' '
                    e2start += 1
                    e2start_idx = token.i
                    continue
                elif (e2start == 1) and (token.text == e2[-1].text) and (token.i>e2start_idx) and (e2end == 0):
                    annotated += token.text + e2_e_tag
                    e2end += 1
                    continue
            annotated += ' ' + token.text + ' '
            
        annotated = annotated.strip()
        annotated = re.sub(' +', ' ', annotated)
        print(annotated)
        return annotated

    def get_annotated_sents_for_test(self,annotated_sent,type1,type2):
        match1 = re.search('\[E1\]', annotated_sent)
        start1 = match1.end()
        match1 = re.search('\[/E1\]', annotated_sent)
        end1 = match1.start()
        match2 = re.search('\[E2\]', annotated_sent)
        start2 = match2.end()
        match2 = re.search('\[/E2\]', annotated_sent)
        end2 = match2.start()
        annotated_for_test = annotated_sent[:start1] + type1 + annotated_sent[end1:start2] + type2 + annotated_sent[end2:]
        return annotated_for_test


    # "cancer can not be treated by dienogest."
    # string "[D]cancer[/D] can not be treated by [C]dienogest[C/]."
    def get_annotated_sents(self, sent):
        sent_nlp = self.nlp(sent)

        pairs = self.get_all_ent_pairs(sent)
        if len(pairs) == 0:
            print('Found less than 2 entities!')
            return
        annotated_list = []
        for pair in pairs:
            annotated = self.annotate_sent(sent_nlp, pair[0], pair[1],pair[2],pair[3])
            print(annotated)

            annotated_list.append([annotated,pair[0], pair[1],pair[2],pair[3]])
        return annotated_list
    
    def get_entity_span(self, x):
        d_start = [i for i, e in enumerate(x) if e == self.D_id][0]
        # d_end =  [i for i, e in enumerate(x) if e == self.d_id_e]
        c_start = [i for i, e in enumerate(x) if e == self.C_id][0]
        # c_end = [i for i, e in enumerate(x) if e == self.c_id_e]


        #e1_e2_start = (e1_start[0],e2_start[0])
        if d_start < c_start:
            e1_e2_start = (d_start,c_start)
        else:
            e1_e2_start = (c_start,d_start)


        # e1_e2_start = ([i for i, e in enumerate(x) if e == self.e1_id][0],\
        #                 [i for i, e in enumerate(x) if e == self.e2_id][0])
        return e1_e2_start

    # "DISEASE can not be treated by CHEMICAL."
    def infer_one_sentence(self, sentence):
        self.net.eval()
        tokenized = self.tokenizer.encode(sentence); #print(tokenized)
        # (1, 6)
        e1_e2_start = self.get_entity_span(tokenized); #print(e1_e2_start)
        tokenized = torch.LongTensor(tokenized).unsqueeze(0)
        e1_e2_start = torch.LongTensor(e1_e2_start).unsqueeze(0)
        attention_mask = (tokenized != self.pad_id).float()
        token_type_ids = torch.zeros((tokenized.shape[0], tokenized.shape[1])).long()
        
        if self.cuda:
            tokenized = tokenized.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            
        classification_logits = self.net(tokenized, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                                    e1_e2_start=e1_e2_start)
        tensor = torch.softmax(classification_logits, dim=1).max(1)[0]

        # if (tensor[0] < 0.6) :
        #     return None
        predicted = torch.softmax(classification_logits, dim=1).max(1)[1].item()

        print("Predicted: ", self.rm.idx2rel[predicted].strip(), '\n')
        return self.rm.idx2rel[predicted].strip()
    
    def infer_sentence(self, sentence, detect_entities=False):
        relations=[]
        #sentences = self.get_annotated_sents(sentence)
        abrv_to_long = dict()
        sent_doc = self.nlp(sentence)
        for abrv in sent_doc._.abbreviations:
            abrv_to_long[str(abrv)] = str(abrv._.long_form)
            print("abrv_to_long:", abrv_to_long)


        sentences_with_paris = self.get_annotated_sents(sentence)
        print('sentences_with_paris: ', sentences_with_paris)

        if sentences_with_paris != None:
            for sentence_with_pair in sentences_with_paris:
                pred ={}
                sent = sentence_with_pair[0]
                sent_for_test = re.sub('\[D\].*\[/D\]','DISEASE',sent)
                sent_for_test = re.sub('\[C\].*\[/C\]','CHEMICAL',sent_for_test)
                if sentence_with_pair[3]=='disease':
                    disease = sentence_with_pair[1]
                    chemical = sentence_with_pair[2]
                else:
                    disease = sentence_with_pair[2]
                    chemical = sentence_with_pair[1]


                if not isinstance(sentence_with_pair[1], list):
                    e1 = sentence_with_pair[1].text
                else:
                    e1 = " ".join([t.text for t in sentence_with_pair[1]])


                if not isinstance(sentence_with_pair[2], list):
                    e2 = sentence_with_pair[2].text
                else:
                    e2 = " ".join([t.text for t in sentence_with_pair[2]])

                relation = self.infer_one_sentence(sent_for_test)

                pred['sent'] =sent
                pred['disease'] = disease
                pred['chemical'] = chemical
                pred['relation'] = relation
                if e1 in abrv_to_long:
                    print("this is a abbreviation:",e1)
                    e1 = e1+'('+abrv_to_long[e1]+')'
                if e2 in abrv_to_long:
                    print("this is a abbreviation:",e2)
                    e2 = e2+'('+abrv_to_long[e2]+')'

                relations.append(pred)
            return relations
