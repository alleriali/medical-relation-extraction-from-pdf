#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from collections import defaultdict
from typing import Dict, List

from . import modeling
from . import optimization
from . import tokenization
from . import tf_metrics

import tensorflow as tf
from tensorflow.python.ops import math_ops
import pickle
import datetime
import spacy
from nltk import word_tokenize

flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "init_checkpoint", '/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/data/pre_trained_ner/model.ckpt-54000',
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)
flags.DEFINE_string(
    "output_dir", '/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/src/BioBERT_NER_RE/output',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 128,  # 384 recommended for longer sentences
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

class InputExample(object):
    """A single example for simple sequence classification."""

    def __init__(self, guid, text):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.guid = guid
        self.text = text


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "[PAD]":
                wf.write(token + '\n')
        wf.close()


def write_tokens_list(tokens, token_test):
    #print('write_tokens_list:', tokens)
    for token in tokens:
        if token != "[PAD]":
            token_test.append(token)


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, token_test):
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    textlist = example.text.split()
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)

    # drop if token is longer than max_seq_length
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)

    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if ex_index < 4:  # Examples before model run
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
    )
    # write_tokens(ntokens,mode)
    write_tokens_list(ntokens, token_test)
    return feature




def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, token_test):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        bert_modified_tokens = []
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, bert_modified_tokens)
        token_test.append(bert_modified_tokens)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = {"predict": tf.argmax(probabilities, axis=-1), "log_probs": log_probs}
        #accuracy = tf.metrics.accuracy(labels=labels, predictions=predict['predict'], name='acc')

        return (predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        #print('model_fn_builder_mode:', mode)
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (predictsDict) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            num_labels, use_one_hot_embeddings)
        predictsDict["input_mask"] = input_mask
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictsDict, scaffold_fn=scaffold_fn
            )
        else:
            tf.logging.fatal("mode != PREDICT")
        return output_spec

    return model_fn

def create_examples_from_sents(sents):
    examples = []
    for (i, sent) in enumerate(sents):
        tokens = word_tokenize(sent)
        sent = " ".join(tokens)
        guid = "%s-%s" % ('test', i)
        text = tokenization.convert_to_unicode(sent)
        examples.append(InputExample(guid=guid, text=text))
    return examples
# "cancer can not be treated by dienogest."

def RunNerOnSentence(sents,
                     bert_dir='/home/ying/PycharmProjects/biobert_weights/biobert_v1.1_pubmed',
                     output_dir='/home/ying/PycharmProjects/medical-relation-extraction-from-pdf/src/BioBERT_NER_RE/output'):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    bert_config = modeling.BertConfig.from_json_file(bert_dir + '/bert_config.json')
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))



    tf.gfile.MakeDirs(output_dir)

    label_list = ["[PAD]", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease", "O", "X", "[CLS]", "[SEP]"]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=bert_dir + '/vocab.txt', do_lower_case=FLAGS.do_lower_case)
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=16, # num_tpu_cores
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    label2id = {}
    for i, label in enumerate(label_list):
        label2id[label] = i
        id2label = {value: key for key, value in label2id.items()}

    token_test = []
    # tokens = word_tokenize(sent)
    # tokenized_sent = " ".join(tokens)
    # predict_examples = []
    # if len(sents)==0:
    #   text = tokenization.convert_to_unicode(tokenized_sent)
    #   predict_examples.append(InputExample(guid=0, text=text))
    predict_examples = create_examples_from_sents(sents)



    predict_file = os.path.join(output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             FLAGS.max_seq_length, tokenizer,
                                             predict_file, token_test)
    tf.logging.error("***** Running prediction*****")
    tf.logging.error("  Num examples = %d", len(predict_examples))
    tf.logging.error("  Batch size = %d", FLAGS.predict_batch_size)
    tf.logging.error("  Example of predict_examples = %s", predict_examples[0].text)
    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    predLabelSents =[]
    pred_results =[]
    for resultIdx, prediction in enumerate(result):
        # Fix for "padding occurrence amid sentence" error
        # (which occasionally cause mismatch between the number of predicted tokens and labels.)
        assert len(prediction["predict"]) == len(prediction[
                                                     "input_mask"]), "len(prediction['predict']) != len(prediction['input_mask']) Please report us!"
        predLabelSent = []
        for predLabel, inputMask in zip(prediction["predict"], prediction["input_mask"]):
            # predLabel : Numerical Value
            if inputMask != 0:
                if predLabel == label2id['[PAD]']:
                    predLabelSent.append('O')
                else:
                    predLabelSent.append(id2label[predLabel])
        predLabelSents.append(predLabelSent)
    #("predLabelSent:", predLabelSent)
    i = 0
    for predLabelSent in predLabelSents:
        pred_lab = []
        for lab in predLabelSent:
            if lab == 'X':
                pred_lab.append('O')
            else:
                pred_lab.append(lab)
        #print("pred_lab:", pred_lab)
        bert_pred = {'toks': [], 'labels': [], 'sentence': []}
        buf = []
        for t, l in zip(token_test[i], pred_lab):
            if t in ['[CLS]', '[SEP]']:  # non-text tokens will not be evaluated.
                bert_pred['toks'].append(t)
                bert_pred['labels'].append(t)  # Tokens and labels should be identical if they are [CLS] or [SEP]
                if t == '[SEP]':
                    bert_pred['sentence'].append(buf)
                    buf = []
                continue
            elif t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
                bert_pred['toks'][-1] += t[2:]  # append pieces to make a full-word
                buf[-1] += t[2:]
            else:
                bert_pred['toks'].append(t)
                bert_pred['labels'].append(l)
                buf.append(t)
        pred_results.append(bert_pred)
        i+=1
    print("NER detokenize done")
    return pred_results


def GetEntities(labels: List[str], tokenized_sentence: List[str], sentence_in_tokens:List[str],nlp):
    sentence = " ".join(sentence_in_tokens[0])
    doc = nlp(sentence)
    token_to_pos_ = dict()
    named_entities = set()
    for ent in doc.ents:
        named_entities.add(ent.text.lower())
    for token in doc:
        token_to_pos_[token.text] = token.pos_
    entity_types = {
        'B-Disease': 'I-Disease',
        'B-Chemical': 'I-Chemical',
    }
    begin = -1
    end = -1
    current_entity_type = ''
    recognized_entities = defaultdict(set)
    recognized_names = defaultdict(set)

    for i in range(len(labels)):
        label = labels[i]
        if current_entity_type != '' and label == entity_types[current_entity_type]:
            end += 1
            continue
        if current_entity_type != '':
            name = ' '.join(tokenized_sentence[begin: end + 1])
            tokens_in_name = end+1-begin
            is_entity = True
            if name in token_to_pos_:
                if tokens_in_name == 1 and token_to_pos_[name] != 'PROPN' and token_to_pos_[name] != 'NOUN':
                    is_entity = False
                else:
                    is_entity = True
            if is_entity and name.lower() not in recognized_names[current_entity_type] and name.lower() not in named_entities and name.find(',')==-1:
                    recognized_entities[current_entity_type].add((begin, end))
                    recognized_names[current_entity_type].add(name.lower())
        if label not in entity_types:
            current_entity_type = ''
            continue
        begin = i
        end = i
        current_entity_type = label
    if current_entity_type != '':
        recognized_entities[current_entity_type].add((begin, end))
    return recognized_entities


def GetPossibleSentences(tokenized_sentence: List[str], entities: Dict, type_1: str, type_2: str,annotated_sents_with_entities:list):

    type_start_map = {
        'B-Disease': '[D]',
        'B-Chemical': '[C]'
    }
    type_end_map = {
        'B-Disease': '[/D]',
        'B-Chemical': '[/C]'
    }
    for entity_1 in entities[type_1]:
        for entity_2 in entities[type_2]:
            positions = []
            inserts = []
            type_1_entity = ' '.join(tokenized_sentence[entity_1[0]: entity_1[1] + 1])
            type_2_entity = ' '.join(tokenized_sentence[entity_2[0]: entity_2[1] + 1])
            if entity_1[0] < entity_2[0]:
                positions = [entity_1[0], entity_1[1], entity_2[0], entity_2[1]]
                inserts = [type_start_map[type_1], type_end_map[type_1], type_start_map[type_2], type_end_map[type_2]]
            else:
                positions = [entity_2[0], entity_2[1], entity_1[0], entity_1[1]]
                inserts = [type_start_map[type_2], type_end_map[type_2], type_start_map[type_1], type_end_map[type_1]]
            sent = tokenized_sentence.copy()
            sent.insert(positions[0], inserts[0])
            sent.insert(positions[1] + 2, inserts[1])
            sent.insert(positions[2] + 2, inserts[2])
            sent.insert(positions[3] + 4, inserts[3])
            sent_with_entity ={'sent':' '.join(sent[1:-1]),'disease':type_1_entity,'chemical':type_2_entity}
            annotated_sents_with_entities.append(sent_with_entity)




def get_annotated_sents_with_entities(sents):
    nlp = spacy.load("en_core_web_sm")
    annotated_sents_with_entities =[]
    pred_results = RunNerOnSentence(sents)
    for pred in pred_results:
        print("pred: ", pred)
        entities = GetEntities(pred['labels'], pred['toks'],pred['sentence'],nlp)
        print("entities",entities)
        GetPossibleSentences(pred['toks'], entities, 'B-Disease', 'B-Chemical', annotated_sents_with_entities)
    return annotated_sents_with_entities

def infer_on_sentence(sent):
    sents = []
    sents.append(sent)
    annotated_sents_with_entities = []
    nlp = spacy.load("en_core_web_sm")
    pred= RunNerOnSentence(sents)[0]
    entities = GetEntities(pred['labels'], pred['toks'], pred['sentence'], nlp)
    extracted_entities = {}
    diseases =[]
    chemicals =[]
    GetPossibleSentences(pred['toks'], entities, 'B-Disease', 'B-Chemical', annotated_sents_with_entities)
    if len(annotated_sents_with_entities)==0:
        for (s,e) in entities['B-Disease']:
            diseases.append(" ".join(pred['toks'][s:e+1]))
        for (s,e) in entities['B-Chemical']:
            chemicals.append(" ".join(pred['toks'][s:e+1]))
        extracted_entities['disease'] = diseases
        extracted_entities['chemical'] = chemicals
        return [annotated_sents_with_entities,extracted_entities]

    return annotated_sents_with_entities