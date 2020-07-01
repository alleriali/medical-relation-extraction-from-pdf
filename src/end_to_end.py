

from BioBERT_NER import ner_lib

result = ner_lib.get_annotated_sents(sent='cancer can not be cancer treated by dienogest.')
print(result)
