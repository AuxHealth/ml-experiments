import os
import json
import argparse
from tqdm import tqdm
from openai import AzureOpenAI
from utils import json_cache, promise, Promise
from metrics.structured_extraction import StructuredExtractionMetric
from dotenv import load_dotenv

OPENAI_MODELS = {
    'gpt35-4k': 'gpt4-subscription',
    'gpt-35-turbo-instruct': 'gpt4-subscription',
    'gpt4-8k': 'gpt4-subscription',
    'gpt-4-turbo': 'gpt4-subscription',
    'gpt-4-turbo-0125': 'auxfinetuning',
}

METRICS = {
    'structured_extraction': StructuredExtractionMetric,
}

@promise
@json_cache(maxsize=80000)
def query_openai(messages,
                 model,
                 azure_subscription,
                 api_version,
                 api_key,
                 max_tokens=1000,
                 temperature=0,
                 top_p=1,
                 frequency_penalty=0,
                 presence_penalty=0):
    client = AzureOpenAI(azure_endpoint=f"https://{azure_subscription}.openai.azure.com/",
                         api_version=api_version,
                         api_key=api_key)
    def func():
        return client.chat.completions.create(
          model=model,
          messages=messages,
          n=1,
          max_tokens=max_tokens,
          temperature=temperature,
          top_p=top_p,
          frequency_penalty=frequency_penalty,
          presence_penalty=presence_penalty,
        )
    response = Promise(func=func).result_with_timeout(timeout=120)
    return response.choices[0].message.content


def main(args):
    if not os.path.exists(args.input_file):
        raise ValueError(f'Data file {args.input_file} does not exist')

    inference_data = [json.loads(line) for line in open(args.input_file)]

    if os.path.exists(args.output_file):
        prediction_data = [json.loads(line) for line in open(args.output_file)]

    else:
        prediction_data = []
        for datum in tqdm(inference_data):
            messages = datum['messages'][:-1]
            response = query_openai(messages=messages,
                                    model=args.model,
                                    azure_subscription=args.azure_subscription,
                                    api_version=args.api_version,
                                    api_key=args.api_key,
                                    max_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    frequency_penalty=args.frequency_penalty,
                                    presence_penalty=args.presence_penalty).result
            prediction_data.append({'messages': messages + [{'role': 'assistant', 'content': response}]})

        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in prediction_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in prediction_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    if args.evaluation_metric == 'structured_extraction':
        metric = StructuredExtractionMetric()
        for pred, target in tqdm(list(zip(prediction_data, inference_data))):
            target_extraction = target['messages'][-1]['content']
            predicted_extraction = pred['messages'][-1]['content']
            metric(target_extraction, predicted_extraction)
        formatted_results = metric.formatted_results()
        print(formatted_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model', type=str, choices=OPENAI_MODELS.keys(), default='gpt35-4k')
    parser.add_argument('--env_file', type=str, default='../.env')
    parser.add_argument('--api_version', type=str, default='2023-12-01-preview')
    parser.add_argument('--evaluation_metric', type=str, default=None, choices=METRICS.keys())

    # Inference parameters
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--frequency_penalty', type=float, default=0)
    parser.add_argument('--presence_penalty', type=float, default=0)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    args.azure_subscription = OPENAI_MODELS[args.model]
    args.api_key = json.loads(os.getenv('AZURE_OPENAI_API_KEYS'))[args.azure_subscription]
    main(args)
