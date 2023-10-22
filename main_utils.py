import os
from utils import predict, postprocess, combine_tokenized_text, get_output_dir, labels_to_ids, combine_text_from_index
import evaluate
import numpy as np
from tqdm import tqdm

seqeval = evaluate.load("seqeval")
import time
import json
from pprint import pprint
from IPython.display import Markdown, display
from IPython.core.display import display, HTML
import os


def gpt3_prediction(eval_dataset, label_list, eval_number=50, is_skip=True):
    from promptify.models.nlp.openai_model import OpenAI
    from promptify.prompts.nlp.prompter import Prompter
    model = OpenAI(api_key="sk-")
    nlp_prompter = Prompter(model)
    true_predictions = []
    true_labels = []
    for eval_data in tqdm(eval_dataset):
        if count > eval_number:
            break
        sent = " ".join(eval_data["tokens"])
        prediction = []
        true_label = [label_list[x] for x in eval_data["ner_tags"]]

        result = nlp_prompter.fit('ner.jinja',
                                  domain='street addresses',
                                  text_input=sent,
                                  labels=label_list)
        try:
            result = eval(result['text'])
            for tokens in result:
                if "T" in tokens:
                    prediction.append(tokens["T"])
            if is_skip and len(prediction) != len(true_label):
                continue
            else:
                while len(prediction) < len(true_label):
                    prediction.append("B-None")
                while len(prediction) > len(true_label):
                    prediction = prediction[:-2]
        except:
            prediction = ["B-None"] * len(true_label)
            time.sleep(1)

        true_predictions.append(prediction)
        true_labels.append(true_label)
    res = seqeval.compute(predictions=true_predictions, references=true_labels)
    return res


def get_latest_checkpoint(model_name):
    output_dir = get_output_dir(model_name)
    model_folder = os.path.join("./", output_dir)
    checkpoints = [d for d in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, d)) and "-" in d]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    latest_folder_path = os.path.join(model_folder, latest_checkpoint)

    return latest_folder_path


def combine_labels(labels):
    result = []
    i = 0
    while i < len(labels):
        label = labels[i]

        if label.startswith("B-"):
            result.append(label[2:])
            i += 1
            while i < len(labels) and labels[i].startswith("I-") and label[2:] == labels[i][2:]:
                i += 1
        else:
            result.append(label[2:])
            i += 1
    return result


def run_predict(sentences, labels, model_name, eval_dataset, label_list, verbose=False):
    none_label = "B-None"
    label_list.append(none_label)
    label_id = {i: label_list[i] for i in range(len(label_list))}
    if model_name == "chatgpt":
        return gpt3_prediction(eval_dataset, label_list)

    test_idx = 10
    text = " ".join(sentences[test_idx])

    model_path = get_latest_checkpoint(model_name)
    pred = predict(text, model_path, eval_dataset)

    if isinstance(pred, list):
        true_predictions = []
        true_labels = []
        for p, tags, token in zip(pred, eval_dataset['ner_tags'], eval_dataset['tokens']):
            try:
                pred_tags = [x['entity'] for x in p]
                tokenized = [postprocess(x['word']) for x in p]
                pred_tags, pred_text = combine_text_from_index(p)
                truth_tags, truth_text = [label_id[x] for x in tags], token
                assert len(pred_tags) == len(truth_tags)
                true_tags = [label_id[x] for x in tags]
                true_tags = combine_labels(true_tags)
                true_predictions.append(pred_tags)
                true_labels.append(truth_tags)
            except Exception as error:
                continue
        res = seqeval.compute(predictions=true_predictions, references=true_labels)
    else:
        tokenized = [postprocess(x['word']) for x in pred]
        pred = [x['entity'] for x in pred]

