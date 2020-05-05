import spacy
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re,os
from itertools import permutations,combinations
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
class PdfToTxt:
    def __init__(self,pdf_path):
        self.nlp = spacy.load('en_core_web_lg')
        txt_path = pdf_path.replace('.pdf','.txt')
        if os.path.exists(txt_path):
            with open (txt_path,'r') as f:
                self.text = f.read()
        else:
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            codec = 'utf-8'
            laparams = LAParams(line_margin=2, char_margin=2)
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
            fp = open(pdf_path, 'rb')
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos = set()
            for i, page in enumerate(PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                                       check_extractable=True)):
                interpreter.process_page(page)
            self.text = retstr.getvalue()
            fp.close()
            device.close()
            retstr.close()
            with open(txt_path, 'w') as f:
                f.write(self.text)



    def get_sub_obj_pairs(self,sent):

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
    def get_processed_sents(self):
        processed_sents  = []
        sents = sent_tokenize(self.text)

        for i,sent in enumerate(sents):
            tokens = word_tokenize(sent)
            if(len(tokens) > 4 and len(tokens)<50): # only take the sents with tokens between 4 and 99

                sent = sent.replace('-\n','').replace('\n',' ').replace('ﬂ ','fl').replace('ﬁ ','fi') # modify typography
                sent = re.sub('(\d{1,4}.){1,}\d{1,4}?',' ',sent)
                match = re.search('\d+[a-z]', sent)
                if match is not None:
                    sent = sent[:match.start()] + sent[match.end() - 1:]
                sent = re.sub('\(\d+.*\)', ' ', sent)
                sent = re.sub(r"\( {0,3}?\d{1,4} {0,3}?[,\-\–]? {0,3}?(\d{1,4})? {0,3}?\)", "", sent)  #remove citations like(12),(1,2)
                sent = re.sub(r"\[ {0,3}?\d{1,4} {0,3}?[,\-\–]? {0,3}?(\d{1,4})? {0,3}?\]", "", sent)  #remove citations like[12],[1-4]
                sent = sent.strip("\n")
                sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
                sent = re.sub("^ +", "", sent) # remove space in front
                sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
                sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
                sent = " ".join([token.replace('-', '') if token.find('[A-Z]') == -1 and token.find('-') != -1 else token for token in
                              word_tokenize(sent)])
                if len(self.get_sub_obj_pairs(sent))>=1: # only take the sents with at least one pair of subject and object
                    processed_sents.append(sent)
        print(len(processed_sents))

        return processed_sents
