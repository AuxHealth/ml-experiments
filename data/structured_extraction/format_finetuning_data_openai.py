import os
import sys
import json
import random
from datetime import datetime, timedelta
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import JinjaTemplateManager


def random_date(start, end):
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    delta = end_date - start_date
    random_days = random.randrange(delta.days + 1)
    random_date = start_date + timedelta(days=random_days)

    random_date_str = random_date.strftime('%Y-%m-%d')
    return random_date_str


def main(args):
    jinja_template_manager = JinjaTemplateManager()

    if not os.path.exists(args.data_file):
        raise ValueError(f'Data file {args.data_file} does not exist')

    formatted_data = []

    data = [json.loads(line) for line in open(args.data_file)]
    if not os.path.exists(args.kb_file):
        raise ValueError(f'KB file {args.kb_file} does not exist')

    kb_data = json.load(open(args.kb_file))
    attribute_definitions = kb_data['attribute_definitions']
    if args.add_kb_examples:
        data.extend(kb_data['prompt_examples']['structured_extraction'])

    for example in data:
        target_symptom = example['metadata']['target_symptom']
        target_attribute = example['metadata']['target_attribute']
        sex = example['metadata']['sex']
        age = example['metadata']['age']
        dr_text = [t['text'] for t in example['data']['dialogue'] if t['speaker'] == 'doctor'][0]
        patient_text = [t['text'] for t in example['data']['dialogue'] if t['speaker'] == 'patient'][0]
        current_date = example['metadata'].get('date', random_date(args.start_date, args.end_date))
        current_day = datetime.strptime(current_date, '%Y-%m-%d').strftime('%A')

        system_prompt = jinja_template_manager.render(
            args.system_prompt_file,
            patient_sex=sex,
            patient_age=age,
            attributes=attribute_definitions)
        user_prompt = jinja_template_manager.render(
            args.user_prompt_file,
            target_symptom=target_symptom,
            target_attribute=target_attribute,
            patient_age=age,
            patient_sex=sex,
            dr_text=dr_text,
            patient_text=patient_text,
            current_day=current_day,
            current_date=current_date)
        target = example['data']['structured_symptoms']
        formatted_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(target)}
            ]
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date for random date generation')
    parser.add_argument('--end_date', type=str, default='2024-02-01', help='End date for random date generation')
    parser.add_argument('--kb_file', type=str, default='../kb/kb_data_20240312.json')
    parser.add_argument('--add_kb_examples', action='store_true', help='Add KB examples (train only)')
    parser.add_argument('--system_prompt_file', type=str, default='structured_extraction_system.jinja')
    parser.add_argument('--user_prompt_file', type=str, default='structured_extraction_user.jinja')

    random.seed(42)
    args = parser.parse_args()
    main(args)
