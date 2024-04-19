import os
import json
import argparse
from tqdm import tqdm
from anthropic import AnthropicVertex
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from utils import json_cache, promise, Promise
from metrics.structured_extraction import StructuredExtractionMetric
from dotenv import load_dotenv

ANTHROPIC_MODELS = {
    'claude-3-sonnet@20240229': {'region': 'us-central1', 'project_id': 'llm-exploration-407519'},
    'claude-3-opus@20240229': {'region': 'us-east5', 'project_id': 'llm-exploration-407519'},
}

METRICS = {
    'structured_extraction': StructuredExtractionMetric,
}


@promise
@json_cache(maxsize=80000)
def query_anthropic(messages,
                    model,
                    gcp_service_account_key,
                    project_id,
                    region,
                    max_tokens=1000,
                    temperature=0,
                    top_p=1,
                    prefill_text=None):
    credentials = service_account.Credentials.from_service_account_info(
        info=gcp_service_account_key,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    if not credentials.valid or credentials.expired:
        credentials.refresh(Request())

    system_prompt = [msg for msg in messages if msg['role'] == 'system'][0]['content']
    messages = [msg for msg in messages if msg['role'] != 'system']
    if prefill_text is not None and isinstance(prefill_text, str):
        messages.append({"role": "assistant", "content": prefill_text})

    client = AnthropicVertex(access_token=credentials.token,
                             project_id=project_id,
                             region=region)

    def func():
        return (client.messages.create(max_tokens=max_tokens,
                                       messages=messages,
                                       model=model,
                                       system=system_prompt,
                                       temperature=temperature,
                                       top_p=top_p))
    response = Promise(func=func).result_with_timeout(timeout=120)

    response_text = response.content[0].text if len(response.content) else ''

    if prefill_text:
        response_text = prefill_text + response_text
    return response_text


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
            response = query_anthropic(messages=messages,
                                       model=args.model,
                                       gcp_service_account_key=args.gcp_service_account_key,
                                       region=args.region,
                                       project_id=args.project_id,
                                       max_tokens=args.max_tokens,
                                       temperature=args.temperature,
                                       top_p=args.top_p,
                                       prefill_text='[').result
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
    parser.add_argument('--model', type=str, choices=ANTHROPIC_MODELS.keys(), default='claude-3-sonnet@20240229')
    parser.add_argument('--env_file', type=str, default='../.env')
    parser.add_argument('--evaluation_metric', type=str, default=None, choices=METRICS.keys())

    # Inference parameters
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--frequency_penalty', type=float, default=0)
    parser.add_argument('--presence_penalty', type=float, default=0)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    args.gcp_service_account_key = json.loads(os.getenv('GCP_SERVICE_ACCOUNT_KEY'))
    args.region = ANTHROPIC_MODELS[args.model]['region']
    args.project_id = ANTHROPIC_MODELS[args.model]['project_id']
    main(args)
