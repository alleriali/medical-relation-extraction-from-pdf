from src.tasks.pdf_to_txt import PdfToTxt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
def get_candidates_for_train(text_path):
    #lemmatizer = WordNetLemmatizer()
    candidate_sents =[]
    key_words = []
    contraindication_key_words=['advice','should not be taken','worse','risk','not to be prescribed',\
                                'contraindicat','should not be prescribed','not use','avoid','exacerbate','caution','careful']
    side_effect_key_words =['side effect','risk','develop','at the cost of','give rise to','have problems with','ill effect',\
                            'associated with the occurrence','trigger','cause','lead to','occur','damage']
    treatment_key_words=['treat','prescribed','for use in','used to','ease','useful','effective','success','reduce','improve',\
                         'relief','relieve','prevent','control','alleviate','suggest']

    Pdf2txt = PdfToTxt(text_path,False)
    processed_sents = Pdf2txt.get_processed_sents()
    for sent in processed_sents:
        candidate_flag = False
        for word in treatment_key_words:
            if word in sent:
                candidate_flag = True
                #candidate_sents.append(sent)    for treatment
                break
        if candidate_flag and ('not' not in sent):
            candidate_sents.append(sent)
    df = pd.DataFrame({'candidate_sents':candidate_sents})
    df.to_csv('/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/Text/candidate_treatment_sents.csv',index=False)
    return candidate_sents


