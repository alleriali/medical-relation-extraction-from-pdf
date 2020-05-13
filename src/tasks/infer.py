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
        else:
            self.nlp = None
        #self.entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
         #                            "WORK_OF_ART", "LAW", "LANGUAGE", 'PER']
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
        
        self.e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        self.e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        self.pad_id = self.tokenizer.pad_token_id
        self.rm = load_pickle("relations.pkl")
        
    def get_all_ent_pairs(self, sent):
        # if isinstance(sent, str):
        #     sent_doc = self.nlp(sent)
        #     sent_ner = self.ner(sent)
        # else:
        #     sents_doc = sent
        sent_doc = self.nlp(sent)
        sent_ner = self.ner(sent)
        ent_text = set()
        diseases = set()
        chemicals = set()
        self.diseases =set()
        # for ent1 in sent_doc.ents:
        #     ent_valid = True
        #     for token in ent1:
        #         if token.tag_ == 'VB':   ## if the ent containing token with tag"verb",this ent should not be included
        #             ent_valid = False
        #             break
        #
        #     if not ent_valid:
        #         continue
        #
        #     for ent2 in sent_ner.ents:
        #         if ent2.text in ent1.text or ent1.text in ent2.text:
        #             if len(ent2.text) >= len(ent1.text):
        #                 ent = ent2
        #             else:
        #                 ent = ent1
        #             if ent.text not in ent_text:
        #                 ent_text.add(ent.text)
        #
        #                 if ent2.label == 9255184837977538312:
        #                     diseases.add(ent)
        #                 else:
        #                     chemicals.add(ent)
        #                 break
        for ent in sent_ner.ents:
            ent_valid = True
            for token in ent:
                if token.tag_=='VB' or token.tag_=='JJ':
                    ent_valid = False
                    break
            if not ent_valid:
                continue
            for ent_in_doc in sent_doc.ents:
                if ent_in_doc.text in ent.text or ent.text in ent_in_doc.text:

                    if ent.text not in ent_text:
                        ent_text.add(ent.text)
                        if ent.label == 9255184837977538312:
                            diseases.add(ent)
                            self.diseases.add(ent.text)
                        else:
                            chemicals.add(ent)
                        break

        print("dieseas:",diseases)
        print("chemicals:",chemicals)

        pairs = set()
        #only consider (disease,disease) and (diesease,chemical) these two relation type
        if len(diseases) >= 1 and len(chemicals)>=1:
            # for a, b in combinations([ent for ent in ents], 2):   ## generate pairs by combinations instead of permutations
            #     pairs.append((a,b))
            for d in diseases:
                for c in chemicals:
                    if d.start<c.start:
                        pairs.add((d,c,'[disease]','[chemical]'))
                    else:
                        pairs.add((c,d,'[chemical]','[disease]'))
            # if len(diseases)>=2:
            #     for a,b in combinations([d for d in diseases],2):
            #         if a.start <b.start:
            #             pairs.add((a,b))
            #         else:
            #             pairs.add((b, a))
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

        print(new_pairs)
        return new_pairs

    def get_all_ents(self, sent):
        if isinstance(sent, str):
            sents_doc = self.ner(sent)
            print("ents")
        else:
            sents_doc = sent
        ents = list(sents_doc.ents)
        print(ents)
        return ents
    
    def get_all_sub_obj_pairs(self, sent):
        if isinstance(sent, str):
            sents_doc = self.nlp(sent)
        else:
            sents_doc = sent
        sent_ = next(sents_doc.sents)
        root = sent_.root
        #print('Root: ', root.text)
        
        subject = None; objs = []; pairs = []
        for child in root.children:
            #print(child.dep_)
            if child.dep_ in ["nsubj", "nsubjpass"]:
                if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                    subject = child; #print('Subject: ', child)
            elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
                objs.append(child); #print('Object ', child)
        
        if (subject is not None) and (len(objs) > 0):
            for a, b in combinations([subject] + [obj for obj in objs], 2):
                a_ = [w for w in a.subtree]
                b_ = [w for w in b.subtree]
                pairs.append((a_[0] if (len(a_) == 1) else a_ , b_[0] if (len(b_) == 1) else b_))
                    
        return pairs
    
    def  annotate_sent(self, sent_nlp, e1, e2):
        annotated = ''
        e1start, e1end, e2start, e2end = 0, 0, 0, 0
        e1start_idx, e2start_idx = 0,0

        for token in sent_nlp:
            if not isinstance(e1, list):
                if (token.text == e1.text) and (e1start == 0) and (e1end == 0):
                    annotated += ' [E1]' + token.text + '[/E1] '
                    e1start, e1end = 1, 1
                    e1start_idx = token.i
                    continue
            else:
                if (token.text == e1[0].text) and (e1start == 0):
                    annotated += ' [E1]' + token.text + ' '
                    e1start += 1
                    e1start_idx = token.i
                    continue
                elif (e1start == 1) and (token.text == e1[-1].text) and (token.i>e1start_idx) and (e1end == 0):
                    print("token i:",token.i)
                    annotated += token.text + '[/E1] '
                    e1end += 1
                    continue
           
            if not isinstance(e2, list):
                if (token.text == e2.text) and (e2start == 0) and (e2end == 0):
                    annotated += ' [E2]' + token.text + '[/E2] '
                    e2start, e2end = 1, 1
                    e2start_idx = token.i
                    continue
            else:
                if (token.text == e2[0].text) and (e2start == 0):
                    annotated += ' [E2]' + token.text + ' '
                    e2start += 1
                    e2start_idx = token.i
                    continue
                elif (e2start == 1) and (token.text == e2[-1].text) and (token.i>e2start_idx) and (e2end == 0):
                    annotated += token.text + '[/E2] '
                    e2end += 1
                    continue
            annotated += ' ' + token.text + ' '
            
        annotated = annotated.strip()
        annotated = re.sub(' +', ' ', annotated)
        print(annotated)
        return annotated

    def get_final_pairs(self,ents,pairs):
        final_pairs = []
        for pair in pairs:
            for ent in ents:
                if isinstance(pair[0],list):
                    pair_text1 = " ".join([token.text for token in pair[0]])
                else:
                    pair_text1 = pair[0].text
                if isinstance(pair[1],list):
                    pair_text2 = " ".join([token.text for token in pair[1]])
                else:
                    pair_text2 = pair[1].text

                if ent.text in pair_text1 or ent.text in pair_text2:
                    print("pair is" ,pair)
                    final_pairs.append(pair)
                    break
        return final_pairs

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


    def get_annotated_sents(self, sent):
        sent_nlp = self.nlp(sent)
        # pairs1 = self.get_all_ent_pairs(sent)
        #ents = self.get_all_ents(sent)
        pairs = self.get_all_ent_pairs(sent)

        #pairs2 = self.get_all_sub_obj_pairs(sent_nlp)
        #print(pairs2)
        #pairs = self.get_final_pairs(ents,pairs2)
        if len(pairs) == 0:
            print('Found less than 2 entities!')
            return
        annotated_list = []
        for pair in pairs:
            # print("type of e1",type(pair[0]))
            # print("type of e2", type(pair[1]))

            annotated = self.annotate_sent(sent_nlp, pair[0], pair[1])
            annotated_for_test = self.get_annotated_sents_for_test(annotated,pair[2],pair[3])
            annotated_list.append([annotated,pair[0], pair[1],annotated_for_test])
        return annotated_list
    
    def get_e1e2_start(self, x):
        e1_start = [i for i, e in enumerate(x) if e == self.e1_id]
        e2_start = [i for i, e in enumerate(x) if e == self.e2_id]
        if len(e1_start)==0 or len(e2_start)==0:
            e1_e2_start =  (-1,-1)
        else:
            e1_e2_start = (e1_start[0],e2_start[0])

        # e1_e2_start = ([i for i, e in enumerate(x) if e == self.e1_id][0],\
        #                 [i for i, e in enumerate(x) if e == self.e2_id][0])
        return e1_e2_start
    
    def infer_one_sentence(self, sentence):
        self.net.eval()
        tokenized = self.tokenizer.encode(sentence); #print(tokenized)
        e1_e2_start = self.get_e1e2_start(tokenized); #print(e1_e2_start)
        if e1_e2_start==(-1,-1):
            return None
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

        if (tensor[0] < 0.6) :
            return None
        predicted = torch.softmax(classification_logits, dim=1).max(1)[1].item()

        print("Predicted: ", self.rm.idx2rel[predicted].strip(), '\n')
        return self.rm.idx2rel[predicted].strip()
    
    def infer_sentence(self, sentence, detect_entities=False):
        if detect_entities:
            preds = {}
            relations=[]
            #sentences = self.get_annotated_sents(sentence)
            abrv_to_long = dict()
            sent_doc = self.nlp(sentence)
            for abrv in sent_doc._.abbreviations:
                abrv_to_long[str(abrv)] = str(abrv._.long_form)
                print("abrv_to_long:", abrv_to_long)


            sentences_with_paris = self.get_annotated_sents(sentence)
            if sentences_with_paris != None:
                for sentence_with_pair in sentences_with_paris:
                    sent = sentence_with_pair[0]
                    #print("annotated_Sent",sent)
                    sent_for_test = sentence_with_pair[3]
                    #print("sent_for_test:",sent_for_test)


                    if not isinstance(sentence_with_pair[1], list):
                        e1 = sentence_with_pair[1].text
                    else:
                        e1 = " ".join([t.text for t in sentence_with_pair[1]])


                    if not isinstance(sentence_with_pair[2], list):
                        e2 = sentence_with_pair[2].text
                    else:
                        e2 = " ".join([t.text for t in sentence_with_pair[2]])

                    pred = self.infer_one_sentence(sent)
                    if pred is None:
                        continue

                    # print("e1:",e1)
                    # print("e2:",e2)
                    # print("self_diseases:",self.diseases)
                    if pred.find('treated_by')!=-1:
                        #print("find treated_by label")
                        # if e1 in self.diseases and e2 in self.diseases:
                        #     #print("both are diseases")
                        #     continue
                        if len(re.findall('\(e2,e1\)', pred)) != 0:
                            if e2 not in self.diseases and e1 in self.diseases:
                                #print("find e1 is disease and e2 is not a disease")
                                pred = 'treated_by(e1,e2)'
                        else:
                            if e1 not in self.diseases and e2 in self.diseases:
                                #print("find e2 is disease and e1 is not a disease")
                                pred = 'treated_by(e2,e1)'
                    preds[sent] = [pred]
                    if e1 in abrv_to_long:
                        print("this is a abbreviation:",e1)
                        e1 = e1+'('+abrv_to_long[e1]+')'
                    if e2 in abrv_to_long:
                        print("this is a abbreviation:",e2)
                        e2 = e2+'('+abrv_to_long[e2]+')'
                    #print("new_prediction:",pred)
                    relations.append([pred,{'e1':e1,'e2':e2,'sent':sent}])
            # print(sentences)
            # if sentences != None:
            #     preds = {}
            #     for sent in sentences:
            #         (sentence, pred) = self.infer_one_sentence(sent)
            #         if (sentence, pred)==(None,None):
            #             continue
            #         preds[sentence] = pred
            #     return preds

                return relations
        else:
            return self.infer_one_sentence(sentence)