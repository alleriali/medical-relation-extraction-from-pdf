import spacy
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re, os, string
from itertools import permutations, combinations
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from tika import parser
def with_pdfminer(pdf_path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams() ## using default parameters line_margin=2, char_margin=2
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(pdf_path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for i, page in enumerate(
            PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                              check_extractable=True)):
        interpreter.process_page(page)
    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text

def with_tika(pdf_path):
    raw = parser.from_file(pdf_path)
    raw_text = str(raw['content'])
    return raw_text

class PdfToTxt:
    def __init__(self, path, is_pdf=False):
        self.nlp = spacy.load('en_core_sci_md')
        if is_pdf:
            pdf_path = path
            txt_path = pdf_path.replace('.pdf', '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    self.text = f.read()
            else:
                self.text = with_tika(pdf_path)
                with open(txt_path, 'w') as f:
                    f.write(self.text)
        else:
            text_path = path
            with open(text_path, 'r') as f:
                self.text = f.read()

    def get_sub_obj_pairs(self, sent):
        if isinstance(sent, str):
            sents_doc = self.nlp(sent)
        else:
            sents_doc = sent
        pairs = []
        has_verb = False
        for token in sents_doc:
            if token.tag_ == "VB":
                has_verb = True
                break
        if has_verb == False:
            return pairs

        try:
            sent_ = next(sents_doc.sents)
        except StopIteration:
            return pairs

        root = sent_.root
        # print('Root: ', root.text)

        subject = None;
        objs = [];
        for child in root.children:
            # print(child.dep_)
            if child.dep_ in ["nsubj", "nsubjpass"]:
                if len(re.findall("[a-z]+", child.text.lower())) > 0:  # filter out all numbers/symbols
                    subject = child;  # print('Subject: ', child)
            elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
                objs.append(child);  # print('Object ', child)

        if (subject is not None) and (len(objs) > 0):
            for a, b in combinations([subject] + [obj for obj in objs], 2):
                a_ = [w for w in a.subtree]
                b_ = [w for w in b.subtree]
                pairs.append((a_[0] if (len(a_) == 1) else a_, b_[0] if (len(b_) == 1) else b_))

        return pairs

    def contain_references(self, sent):
        tokens = word_tokenize(sent)
        count = 0
        token_istitle = [token.istitle() for token in tokens]
        type_istitle = [True, True, False]
        sent_size = len(token_istitle)
        for index, flag in enumerate(token_istitle):
            if type_istitle[0] == flag and index < sent_size-2:
                if type_istitle[1] == token_istitle[index + 1] and type_istitle[2] == token_istitle[index + 2]:
                    count += 1
        if count >= 2:
            return True
        else:
            return False

    def no_spaces_between_words(self, tokens):
        for token in tokens:
            if len(token) > 46:
                return True
        return False



    def remove_title(self,sent):
        tokens = word_tokenize(sent)

        for idx, token in enumerate(tokens):
            if token.isupper() and token.isalpha():
                words_and_numbers = set()
                i = idx
                start_of_title, end_of_title = 0, 0
                while i >= 0:
                    if tokens[i].isupper() and tokens[i].isalpha() or tokens[i].isnumeric():

                        words_and_numbers.add(tokens[i])
                        i -= 1
                    elif tokens[i] in string.punctuation:
                        start_of_title = i
                        break
                    else:
                        start_of_title = i + 1
                        break

                j = idx
                while j < len(tokens):
                    if tokens[j].isupper() and tokens[j].isalpha() or tokens[j].isnumeric():

                        words_and_numbers.add(tokens[j])
                        j += 1
                    elif tokens[j] in string.punctuation:
                        end_of_title = j
                        break
                    else:
                        end_of_title = j - 1
                        break

                if len(
                        words_and_numbers) > 1 and end_of_title - start_of_title == 1 or end_of_title - start_of_title > 1:
                    return " ".join(tokens[:start_of_title] + tokens[end_of_title + 1:])
        return " ".join(tokens)



    def get_processed_sents(self):
        processed_sents = []
        text = self.text
        text = text.replace('-\n', '').replace('ﬂ ', 'fl').replace('ﬁ ','fi')  # replace break line and modify typography
        text = " ".join(text.splitlines())  # delete all line breaks
        text = text.replace('\t', '')  # delete table key
        text = re.sub('[\u2022,\u2023,\u25E6,\u2043,\u2219]',',',text) #  replace common bullet symbols(Bullet (•) Triangular Bullet (‣) White Bullet (◦) Hyphen Bullet (⁃) Bullet Operator (∙) with full stop
        text = re.sub(r'(\s)(\d+|[a-z])(\.\s+[A-Z])', r'\3', text) # delete the number in number lists , for example, replace 3. with .
        text = re.sub(r'®','',text)
        text = re.sub(r'([a-z]+)([0-9]+)',r'\1',text)
        #text = re.sub(r"\( {0,3}?Fig {0,3}?.[\s\dA-Z]+\)", "", text)  # delete the figure citation
        sents = sent_tokenize(text)
        for sent in sents:
            tokens = word_tokenize(sent)
            if self.no_spaces_between_words(tokens):
                continue
            sent_length = len(tokens)
            if sent_length >= 4 and sent_length <=40 :

                sent = re.sub(r"\[.*\]", "", sent)  # delete all [ ]
                sent = re.sub(r"\([\d\s,]+\)", "", sent)  # delete citations like ( 2,3)
                sent = re.sub(r"^[\d.]+", "", sent)  # delete the serial numbers at the beginning of the sentence
                sent = sent.strip(" ")

                if self.contain_references(sent):  # delete the sentences mixed with references
                    continue
                tokens = word_tokenize(sent)
                if len(tokens)<2:
                    continue
                if tokens[0].isupper() and tokens[0].isalpha() and tokens[1].istitle():
                    sent = " ".join(tokens[1:])
                # if len(re.findall(r"\d+\s+[A-Z]{3,}", sent)) != 0:  # delete the sentences mixed with numbers and headings
                #     continue
                # if len(re.findall(r"[A-Z]+\s+\d+", sent)) != 0:  # delete the sentences mixed with headings and numbers
                #     continue

                sent = self.remove_title(sent)
                if len(self.get_sub_obj_pairs(sent)) >= 1:  # only take the sents with at least one pair of subject and object and the sentence contains at least one verb
                    processed_sents.append(sent)
        return processed_sents

        # self.text = re.sub('[●•]', '.', self.text)
        # sents = sent_tokenize(self.text)
        #
        # for i,sent in enumerate(sents):
        #     tokens = word_tokenize(sent)
        #     if(len(tokens) > 4 and len(tokens)<50): # only take the sents with tokens between 4 and 99
        #
        #         sent = sent.replace('-\n','').replace('\n',' ').replace('ﬂ ','fl').replace('ﬁ ','fi') # modify typography
        #         sent = re.sub('[—`*]', '', sent)
        #         sent = re.sub('(\d{1,4}.){1,}\d{1,4}?',' ',sent)
        #         match = re.search('\d+[a-z]', sent)
        #         if match is not None:
        #             sent = sent[:match.start()] + sent[match.end() - 1:]
        #         sent = re.sub('\(\d+.*\)', ' ', sent)
        #
        #
        #         sent = re.sub(r"\( {0,3}?\d{1,4} {0,3}?[,\-\–]? {0,3}?(\d{1,4})? {0,3}?\)", "", sent)  #remove citations like(12),(1,2)
        #         sent = re.sub(r"\[ {0,3}?\d{1,4} {0,3}?[,\-\–]? {0,3}?(\d{1,4})? {0,3}?\]", "", sent)  #remove citations like[12],[1-4]
        #         sent = sent.strip("\n")
        #         sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
        #         sent = re.sub("^ +", "", sent) # remove space in front
        #         sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
        #         sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
        #         sent = " ".join([token.replace('-', '') if token.find('[A-Z]') == -1 and token.find('-') != -1 else token for token in
        #                       word_tokenize(sent)])
        #         sent = re.sub('—', '', sent)
        #         if len(self.get_sub_obj_pairs(sent))>=1: # only take the sents with at least one pair of subject and object
        #             processed_sents.append(sent)
        # print(len(processed_sents))
