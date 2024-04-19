import os
import json
from urllib.request import Request, urlopen
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import dotenv
dotenv.load_dotenv('../.env')

def query_azure_embeddings(texts_to_query,
                           url=os.getenv('AZURE_SAPBERT_ENDPOINT_URL'),
                           api_key=os.getenv('AZURE_SAPBERT_ENDPOINT_KEY'),
                           deployment_name='sapbert-inference-1'):
    body = str.encode(json.dumps({'query': texts_to_query}))
    headers = {
        'Content-Type': 'application/json',
        'Authorization': ('Bearer ' + api_key),
        'azureml-model-deployment': deployment_name
    }
    request = Request(url, body, headers)
    response = urlopen(request)
    result = response.read()
    queried_embeddings = json.loads(result.decode())
    return queried_embeddings


def check_format(pred: dict):
    '''
    Check if dict is of format:
    {
        'name': <name of symptom>,
        'attributes': {
            'presence': <present, absent or unknown>,
            <attribute name>: [<attribute value1>, <attribute value2>, ...],
            ...,
        }
    }
    '''
    return all([('name' in symptom_attributes.keys()
                and 'attributes' in symptom_attributes.keys() and isinstance(symptom_attributes['attributes'], dict)
                and 'presence' in symptom_attributes['attributes'].keys() 
                and symptom_attributes['attributes']['presence'] in ['present', 'absent']) for symptom_attributes in pred])


def extract_outermost_braces(s):
    start_index = s.find('[')
    end_index = s.rfind(']')
    if start_index == -1 or end_index == -1:
        return ''
    else:
        return s[start_index: end_index + 1]


def embedding_evaluation(prediction: list[str], target: list[str], threshold=0.8):
    if len(prediction) == 0:
        return 0, 0, len(target)
    if len(target) == 0:
        return 0, len(prediction), 0
    if len(prediction) and len(target):
        prediction_embeddings = query_azure_embeddings(prediction)
        target_embeddings = query_azure_embeddings(target)
        similarity_matrix = cosine_similarity(prediction_embeddings, target_embeddings)

        # Determine matches based on threshold
        predicted_matches = (similarity_matrix > threshold).sum(axis=1)
        target_matches = (similarity_matrix > threshold).sum(axis=0)

        # For computing precision, recall, and F1 score
        tp = np.sum(predicted_matches > 0)  # True positives: Predicted embeddings with at least one match in target embeddings
        fp = np.sum(predicted_matches == 0)  # False positives: Predicted embeddings with no matches in target embeddings
        fn = np.sum(target_matches == 0)    # False negatives: Target embeddings with no matches in predicted embeddings
    elif len(prediction):
        tp = 0
        fp = len(prediction)
        fn = 0
    elif len(target):
        tp = 0
        fp = 0
        fn = len(target)
    return tp, fp, fn


class StructuredExtractionMetric(object):
    def __init__(self):
        self.examples = 0
        self.evaluated = 0
        self.symptoms_results = {'fp': 0, 'fn': 0, 'tp': 0}
        self.attribute_name_results = {'fp': 0, 'fn': 0, 'tp': 0}
        self.attribute_value_results = {'fp': 0, 'fn': 0, 'tp': 0}

    def formatted_results(self):
        result_string = ""

        result_string += f"Evaluated: {self.evaluated / self.examples}\n"

        if (self.symptoms_results['tp'] + self.symptoms_results['fp']) > 0:
            symptom_precision = self.symptoms_results['tp'] / (self.symptoms_results['tp'] + self.symptoms_results['fp'])
        else:
            symptom_precision = 0

        if (self.symptoms_results['tp'] + self.symptoms_results['fn']) > 0:
            symptom_recall = self.symptoms_results['tp'] / (self.symptoms_results['tp'] + self.symptoms_results['fn'])
        else:
            symptom_recall = 0
        result_string += f"Symptoms precision: {symptom_precision}, recall: {symptom_recall}\n"

        if (self.attribute_name_results['tp'] + self.attribute_name_results['fp']) > 0:
            attr_name_precision = self.attribute_name_results['tp'] / (self.attribute_name_results['tp'] + self.attribute_name_results['fp'])
        else:
            attr_name_precision = 0
        if (self.attribute_name_results['tp'] + self.attribute_name_results['fn']) > 0:
            attr_name_recall = self.attribute_name_results['tp'] / (self.attribute_name_results['tp'] + self.attribute_name_results['fn'])
        else:
            attr_name_recall = 0
        result_string += f"Attributes name precision: {attr_name_precision}, recall: {attr_name_recall}\n"

        if (self.attribute_value_results['tp'] + self.attribute_value_results['fp']) > 0:
            attr_value_precision = self.attribute_value_results['tp'] / (self.attribute_value_results['tp'] + self.attribute_value_results['fp'])
        else:
            attr_value_precision = 0
        if (self.attribute_value_results['tp'] + self.attribute_value_results['fn']) > 0:
            attr_value_recall = self.attribute_value_results['tp'] / (self.attribute_value_results['tp'] + self.attribute_value_results['fn'])
        else:
            attr_value_recall = 0
        result_string += f"Attributes value precision: {attr_value_precision}, recall: {attr_value_recall}"
        return result_string

    def __call__(self, reference, prediction):
        self.examples += 1
        prediction_cleaned = extract_outermost_braces(prediction)
        try:
            prediction = json.loads(prediction_cleaned)
        except Exception:
            return self.formatted_results()

        if not check_format(prediction):
            return self.formatted_results()

        if isinstance(reference, str):
            reference = json.loads(reference)

        # Symptom name evaluation
        gold_symptoms = set([symp['name'] for symp in reference if 'name' in symp])
        predicted_symptoms = set([symp['name'] for symp in prediction if 'name' in symp])
        tp, fp, fn = embedding_evaluation(list(predicted_symptoms), list(gold_symptoms))
        self.symptoms_results['fp'] += fp
        self.symptoms_results['fn'] += fn
        self.symptoms_results['tp'] += tp

        # Attribute name evaluation
        for symptom in reference:
            if symptom['name'] in predicted_symptoms:
                pred_symptom = [s for s in prediction if s['name'] == symptom['name']][0]
                gold_attribute_names = [name for name in symptom['attributes'].keys()]
                pred_attribute_names = [name for name in pred_symptom['attributes'].keys()]
                if len(gold_attribute_names) and len(pred_attribute_names):
                    tp, fp, fn = embedding_evaluation(pred_attribute_names, gold_attribute_names)
                    self.attribute_name_results['tp'] += tp
                    self.attribute_name_results['fp'] += fp
                    self.attribute_name_results['fn'] += fn
                elif len(gold_attribute_names):
                    self.attribute_name_results['fn'] += len(gold_attribute_names)
                elif len(pred_attribute_names):
                    self.attribute_name_results['fp'] += len(pred_attribute_names)
            else:
                self.attribute_name_results['fn'] += (len(symptom['attributes'].keys()) - 1)
        for symptom in prediction:
            if symptom['name'] not in gold_symptoms:
                self.attribute_name_results['fp'] += (len(symptom['attributes'].keys()) - 1)

        # Attribute values evaluation
        gold_attribute_values = set()
        for symptom in reference:
            if 'attributes' in symptom.keys():
                for attr_values in symptom['attributes'].values():
                    if isinstance(attr_values, list):
                        for val in attr_values:
                            gold_attribute_values.add(val)

        pred_attribute_values = set()
        for symptom in prediction:
            if 'attributes' in symptom.keys() and isinstance(symptom['attributes'], dict):
                for attr_values in symptom['attributes'].values():
                    if isinstance(attr_values, list):
                        for val in attr_values:
                            pred_attribute_values.add(val)
        tp, fp, fn = embedding_evaluation(list(pred_attribute_values), list(gold_attribute_values))
        self.attribute_value_results['fp'] += fp
        self.attribute_value_results['fn'] += fn
        self.attribute_value_results['tp'] += tp
        self.evaluated += 1
        return self.formatted_results()
