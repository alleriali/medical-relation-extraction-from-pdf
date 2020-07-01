import PyPDF2,os
from tika import parser
import pandas as pd
import fitz
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import spacy
import glob
from src.tasks.pdf_to_txt import PdfToTxt
nlp = spacy.load("en_core_sci_md")
ner = spacy.load("en_ner_bc5cdr_md")

def get_all_ent_pairs(sent):

    sent_doc = nlp(sent)
    sent_ner = ner(sent)
    ent_text = set()
    chemicals = set()
    diseases = set()

    for ent in sent_ner.ents:
        ent_valid = True
        for token in ent:
            if token.tag_ == 'VB' or token.tag_ == 'JJ':
                ent_valid = False
                break
        if not ent_valid:
            continue
        for ent_in_doc in sent_doc.ents:
            if ent_in_doc.text in ent.text or ent.text in ent_in_doc.text:

                if ent.text not in ent_text:
                    ent_text.add(ent.text)
                    if ent.label_ == "DISEASE":
                        diseases.add(ent)
                    else:
                        chemicals.add(ent)
                    break


    pairs = set()
    # only consider (disease,disease) and (diesease,chemical) these two relation type
    if len(diseases) >= 1 and len(chemicals) >= 1:
        # for a, b in combinations([ent for ent in ents], 2):   ## generate pairs by combinations instead of permutations
        #     pairs.append((a,b))
        for d in diseases:
            for c in chemicals:
                if d.start < c.start:
                    pairs.add((d, c))
                else:
                    pairs.add((c, d))

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
        new_pairs.append((p0, p1))

    return new_pairs


def with_pypdf2(pdf_path):
    pdf = os.path.basename(pdf_path)
    print(pdf)
    txt = 'with_pypdf2_'+pdf.replace('.pdf', '.txt')
    output_path = '/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/Text/text_from_toolkits/'+txt
    if os.path.exists(output_path):
        return output_path
    pdfFileObject = open(pdf_path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
    count = pdfReader.numPages
    text =""
    for i in range(count):
        page = pdfReader.getPage(i)
        text+= page.extractText()
    with open(output_path,'w') as f:
        f.write(text)
    return output_path


def with_pdfminer(pdf_path):
    pdf = os.path.basename(pdf_path)
    print(pdf)
    txt = 'with_pdfminer_' + pdf.replace('.pdf', '.txt')
    output_path = '/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/Text/text_from_toolkits/'+txt
    if os.path.exists(output_path):
        return output_path
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
    with open(output_path,'w') as f:
        f.write(text)
    return output_path

def with_tika(pdf_path):
    pdf = os.path.basename(pdf_path)
    print(pdf)
    txt = 'with_tika_' + pdf.replace('.pdf', '.txt')
    output_path = '/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/Text/text_from_toolkits/'+txt
    if os.path.exists(output_path):
        return output_path
    raw = parser.from_file(pdf_path)
    raw = str(raw['content'])
    with open(output_path,'w') as f:
        f.write(raw)
    return output_path

def with_pymuPDF(pdf_path):
    pdf = os.path.basename(pdf_path)
    print(pdf)
    txt = 'with_pymuPDF_' + pdf.replace('.pdf', '.txt')
    output_path = '/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/Text/text_from_toolkits/' + txt
    if os.path.exists(output_path):
        return output_path
    doc = fitz.open(pdf_path)
    page_count = doc.pageCount
    page = 0
    text = ''
    while (page < page_count):
        p = doc.loadPage(page)
        text += p.getText()
        page += 1
    with open(output_path,'w') as f:
        f.write(text)
    return output_path


# if __name__ == "__main__":
#     base = '/home/ying/PycharmProjects/BERT-Relation-Extraction/data/PDF1/'
#     pdfs = ['Carson_Paediatric_Dermatology_2nd.pdf', 'Duhring_Diseases_of_the_skin.pdf',
#             'Kane_Paediatric_Dermatology.pdf',
#             'Hartzell_Diseases_of_the_skin.pdf', 'Hurwitz_Pediatric_Dermatology_5th.pdf',
#             'Arenas_Tropical_Dermatology.pdf']
#     functions = [with_pypdf2, with_pdfminer, with_tika, with_pymuPDF]
#     records = dict()
#     record_pairs = dict()
#     for pdf in pdfs:
#         count_sents = []
#         count_pairs =[]
#         for func in functions:
#             pdf_path = base + pdf
#             text_path = func(pdf_path)
#             pdf2text = PdfToTxt(text_path, is_pdf=False)
#             processed_sents = pdf2text.get_processed_sents()
#             file_path = '/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/processed_sents/' + str(
#                 func) + '_' + pdf.replace('.pdf', '.txt')
#             with open(file_path, 'w') as f:
#                 for sent in processed_sents:
#                     f.write(sent + '\n')
#             candidate_sents= 0
#             for sent in processed_sents:
#                 pairs = get_all_ent_pairs(sent)
#                 if len(pairs)>=1:
#                     candidate_sents+=1
#             count_sents.append(len(processed_sents))
#             count_pairs.append(candidate_sents)
#         records[pdf] = count_sents
#         record_pairs[pdf] = count_pairs
#     record_df = pd.DataFrame(records, index=['with_pypdf2', 'with_pdfminer', 'with_tika', 'with_pymuPDF'])
#     record_pairs_df = pd.DataFrame(record_pairs, index=['with_pypdf2', 'with_pdfminer', 'with_tika', 'with_pymuPDF'])
#     record_df.to_csv('./data/record_df_6pdf.csv')
#     record_pairs_df.to_csv(('./data/record_pair_6pdf.csv'))

if __name__ == "__main__":
    files = []
    sents_numbers =[]
    for file in glob.glob('/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/processed_sents/*.txt'):
        file_name = os.path.basename(file)
        files.append(file_name)
        with open(file, 'r') as f:
            processed_sents = f.readlines()
            candidate_sents = 0
            for sent in processed_sents:
                pairs = get_all_ent_pairs(sent)
                if len(pairs) >= 1:
                    candidate_sents += 1
            sents_numbers.append(candidate_sents)
    record_pairs_df = pd.DataFrame({'file_names':files,'sents_number':sents_numbers})

    record_pairs_df.to_csv(('./data/record_candidate_sents.csv'))


