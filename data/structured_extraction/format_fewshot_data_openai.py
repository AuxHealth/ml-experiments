import os
import sys
import json
from urllib.request import Request, urlopen
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime, timedelta
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import JinjaTemplateManager


def query_azure_embeddings(url, api_key, deployment_name, texts_to_query):
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


def random_date(start, end):
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    delta = end_date - start_date
    random_days = random.randrange(delta.days + 1)
    random_date = start_date + timedelta(days=random_days)

    random_date_str = random_date.strftime('%Y-%m-%d')
    return random_date_str


def main(args):
    if not os.path.exists(args.demonstration_data_file):
        raise ValueError(f'Data file {args.demonstration_data_file} does not exist')

    if not os.path.exists(args.inference_data_file):
        raise ValueError(f'Data file {args.inference_data_file} does not exist')

    if not os.path.exists(args.kb_file):
        raise ValueError(f'KB file {args.kb_file} does not exist')

    kb_data = json.load(open(args.kb_file))
    attribute_definitions = kb_data['attribute_definitions']

    jinja_template_manager = JinjaTemplateManager()

    demonstration_data = [json.loads(line) for line in open(args.demonstration_data_file)]
    demonstration_text = [example['data']['dialogue'][-1]['text'] for example in demonstration_data]

    cache_file = os.path.join(args.cache_path, f'{args.deployment_name}_{args.deployment_version}.json')
    if os.path.exists(cache_file):
        embedding_cache = json.load(open(cache_file))
    else:
        embedding_cache = {}

    source_embeddings = []
    for text in tqdm(demonstration_text):
        embedding = embedding_cache.get(text, None)
        if embedding is None:
            embedding = query_azure_embeddings(args.url,
                                               args.api_key,
                                               args.deployment_name,
                                               [text])
        source_embeddings.append(embedding[0])
        embedding_cache[text] = embedding

    with open(cache_file, 'w') as f:
        json.dump(embedding_cache, f)

    source_embeddings = np.array(source_embeddings)
    inference_data = [json.loads(line) for line in open(args.inference_data_file)]
    formatted_data = []
    for example in tqdm(inference_data):
        system_prompt = jinja_template_manager.render('structured_data_instruction.jinja',
                                                      patient_sex=example['metadata']['sex'],
                                                      patient_age=example['metadata']['age'],
                                                      attributes=attribute_definitions)
        messages = [{'role': 'system', 'content': system_prompt}]

        # Add the top 5 most similar demonstrations
        query_text = example['data']['dialogue'][-1]['text']
        embedding = embedding_cache.get(query_text,
                                        query_azure_embeddings(args.url,
                                                               args.api_key,
                                                               args.deployment_name,
                                                               [query_text]))
        sim = cosine_similarity(np.array(embedding), source_embeddings)[0]
        demonstrations = [demonstration_data[idx] for idx in np.argsort(sim)[-5:]]
        for demo in demonstrations:
            current_date = demo['metadata'].get('date', random_date(args.start_date, args.end_date))
            current_day = datetime.strptime(current_date, '%Y-%m-%d').strftime('%A')
            demonstration_prompt = jinja_template_manager.render(
                'structured_data_example.jinja',
                target_attribute=demo['data']['target_attribute'],
                target_symptom=demo['data']['target_symptom'],
                dr_text=demo['data']['dialogue'][0]['text'],
                patient_text=demo['data']['dialogue'][1]['text'],
                patient_age=demo['metadata']['age'],
                patient_sex=demo['metadata']['sex'],
                current_day=current_day,
                current_date=current_date
            )
            messages.append({'role': 'user', 'content': demonstration_prompt})
            messages.append({'role': 'assistant', 'content': json.dumps(demo['data']['structured_symptoms'])})

        # Add the current example
        current_date = example['metadata'].get('date', random_date(args.start_date, args.end_date))
        current_day = datetime.strptime(current_date, '%Y-%m-%d').strftime('%A')
        user_prompt = jinja_template_manager.render(
            'structured_data_example.jinja',
            target_attribute=example['data']['target_attribute'],
            target_symptom=example['data']['target_symptom'],
            dr_text=example['data']['dialogue'][0]['text'],
            patient_text=example['data']['dialogue'][1]['text'],
            patient_age=example['metadata']['age'],
            patient_sex=example['metadata']['sex'],
            current_day=current_day,
            current_date=current_date
        )
        messages.append({'role': 'user', 'content': user_prompt})

        # Also append annotated target
        messages.append({'role': 'assistant', 'content': json.dumps(example['data']['structured_symptoms'])})
        formatted_data.append({'messages': messages})

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demonstration_data_file', type=str, default='data/training_turns_11435.jsonl')
    parser.add_argument('--inference_data_file', type=str, default='data/structured_extraction_test_1000.jsonl')
    parser.add_argument('--output_file', type=str, default='data/openai_fewshot_structured_extraction_test_1000.jsonl')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date for random date generation')
    parser.add_argument('--end_date', type=str, default='2024-02-01', help='End date for random date generation')
    parser.add_argument('--kb_file', type=str, default='../kb/kb_data_20240312.json')
    parser.add_argument('--env_file', type=str, default='../../.env')
    parser.add_argument('--cache_path', type=str, default='.cache')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--system_prompt_file', type=str, default='structured_extraction_system.jinja')
    parser.add_argument('--user_prompt_file', type=str, default='structured_extraction_user.jinja')
    parser.add_argument('--n_shot', type=int, default=5)
    args = parser.parse_args()
    load_dotenv(args.env_file)

    args.api_key = os.getenv('AZURE_PUBMEDBERT_ENDPOINT_KEY')
    args.url = os.getenv('AZURE_PUBMEDBERT_ENDPOINT_URL')
    args.deployment_name = "pubmedbert-deployment"
    args.deployment_version = os.getenv('AZURE_PUBMEDBERT_EMBEDDING_VERSION')

    main(args)
