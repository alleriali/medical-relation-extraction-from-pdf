# medical-relation-extraction-from-pdf

After download, run the file main.py, you'll extract medical relations from the PDFs in the folder data/PDF_FOR_PARSE

Then you could search for the relations by giving a relation type or entity names, a graph containing the output relations will be shown on.

Except for the searching for relations stored in the knowledge graph, you can also try infer function: by giving a sentence, you can get the 
entities and corresponding relations between the entities.

Except main.py, you can also run scispacy_with_entity_position_in_model.py to get the same functions introduced above.


scispacy_with_entity_position_in_model : scispacy NER + relation classifier based on the relation representation is the concatanation of the final hidden 
states of DISEASE and CHEMICAL. The training of relation classifier is referred from https://github.com/plkmo/BERT-Relation-Extraction.

main.py: fine tune BioBERT on NER and relation extraction tasks, the code for training is referred from https://github.com/dmis-lab/biobert.




