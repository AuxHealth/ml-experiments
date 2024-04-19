import os
import json
import argparse
from tqdm import tqdm
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from utils import json_cache, promise, Promise
from metrics.structured_extraction import StructuredExtractionMetric
from google.cloud.aiplatform_v1beta1.types.content import HarmCategory, SafetySetting
from dotenv import load_dotenv

VERTEX_MODELS = {
    'gemini-1.5-pro-preview-0409': {'region': 'us-central1', 'project_id': 'llm-exploration-407519'},
}

METRICS = {
    'structured_extraction': StructuredExtractionMetric,
}


@promise
@json_cache(maxsize=80000)
def query_vertex(messages,
                 model,
                 gcp_service_account_key,
                 project_id,
                 region,
                 max_tokens=1000,
                 temperature=0,
                 top_p=1):
    credentials = service_account.Credentials.from_service_account_info(
        info=gcp_service_account_key,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    if not credentials.valid or credentials.expired:
        credentials.refresh(Request())
    vertexai.init(project=project_id, location=region, credentials=credentials)

    contents = []
    for msg in messages:
        if msg['role'] == 'user':
            contents.append({'role': 'user', 'parts': [{'text': msg['content']}]})
        elif msg['role'] == 'assistant':
            contents.append({'role': 'model', 'parts': [{'text': msg['content']}]})

    system_instruction = [msg for msg in messages if msg['role'] == 'system'][0]['content']
    generation_config = {
        'temperature': temperature,
        'top_p': top_p,
        'candidate_count': 1,
        'max_output_tokens': max_tokens,
    }
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    model = GenerativeModel(model, system_instruction=system_instruction)
    model_input = {
        'contents': contents,
        'generation_config': generation_config,
        'safety_settings': safety_settings
    }

    def func():
        return (model.generate_content(**model_input))

    response = Promise(func=func).result_with_timeout(timeout=120)
    try:
        result = response.candidates[0].content.parts[0].text
    except:
        result = "Failed Filtering Content"

    return result


def main(args):
    if not os.path.exists(args.input_file):
        raise ValueError(f'Data file {args.input_file} does not exist')

    inference_data = [json.loads(line) for line in open(args.input_file)]

    if os.path.exists(args.output_file):
        prediction_data = [json.loads(line) for line in open(args.output_file)]

    else:
        prediction_data = []
        for idx, datum in enumerate(tqdm(inference_data)):
            messages = datum['messages'][:-1]
            response = query_vertex(messages=messages,
                                    model=args.model,
                                    gcp_service_account_key=args.gcp_service_account_key,
                                    region=args.region,
                                    project_id=args.project_id,
                                    max_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p).result
            prediction_data.append({'messages': messages + [{'role': 'assistant', 'content': response}]})

            # if (idx + 1) % 5 == 0:
            #     import time
            #     time.sleep(60)

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
    parser.add_argument('--model', type=str, choices=VERTEX_MODELS.keys(), default='gemini-1.5-pro-preview-0409')
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
    args.region = VERTEX_MODELS[args.model]['region']
    args.project_id = VERTEX_MODELS[args.model]['project_id']
    main(args)
