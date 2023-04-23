import os
import argparse
from align import aligner
import pandas as pd
import json

parser = argparse.ArgumentParser(description='Get configurations to train')
# parser.add_argument('--accent', default='orig', type=str)
parser.add_argument('--lang', default='en', type=str)
# parser.add_argument('--type', default='train', type=str)
CONFIG = parser.parse_args()

accents = {
    'en': ['tamil_female', 'bengali_female', 'malayalam_male', 'manipuri_female', 'assamese_female', 'gujarati_male', 'telugu_male', 'kannada_male', 'hindi_female', 'rajasthani_female', 'kannada_female', 'bengali_male', 'tamil_male', 'gujarati_female', 'assamese_male'],
    'hi': ['bengali_female', 'bengali_male', 'gujarati_female', 'gujarati_male', 'hindi_female', 'hindi_male', 'kannada_female', 'kannada_male', 'malayalam_female', 'malayalam_male', 'manipuri_female', 'rajasthani_female', 'rajasthani_male', 'tamil_female', 'tamil_male', 'telugu_female', 'telugu_male']
}

splits = ['train', 'validation', 'test']

dir = f'../dataset/massive/{CONFIG.lang}'


# data = {
#     "0": {
#         "id": "0",
#         "locale": "en-US",
#         "partition": "test",
#         "scenario": 16,
#         "intent": 48,
#         "utt": " Wait me up at 5 am this week.",
#         "annot_utt": "wake me up at [time : five am] of maybe [date : this week]",
#         "worker_id": "1",
#         "slot_method": {
#             "slot": [],
#             "method": []
#         },
#         "judgments": {
#             "worker_id": [],
#             "intent_score": [],
#             "slots_score": [],
#             "grammar_score": [],
#             "spelling_score": [],
#             "language_identification": []
#         }
#     },
#     "10": {
#         "id": "50",
#         "locale": "en-US",
#         "partition": "test",
#         "scenario": 5,
#         "intent": 38,
#         "utt": " All tell me the time in a ''M'' mean plus five.",
#         "annot_utt": "olly tell me the time in [time_zone : g. m. t. plus five]",
#         "worker_id": "1",
#         "slot_method": {
#             "slot": [],
#             "method": []
#         },
#         "judgments": {
#             "worker_id": [],
#             "intent_score": [],
#             "slots_score": [],
#             "grammar_score": [],
#             "spelling_score": [],
#             "language_identification": []
#         }
#     }
# }

# _, aligned_slot_labels = aligner(data, multiple=True)
# for (key, entry), slot_label in zip(data.items(), aligned_slot_labels):
#     entry["slot_labels"] = slot_label
# with open('temp.json', "w+", encoding="utf-8") as file:
#     json.dump(data, file, ensure_ascii=False)

for accent in accents[CONFIG.lang]:
    for split in splits:
        file_name = f'{dir}/{accent}/{split}.json'
        data = pd.read_json(file_name).to_dict()
        _, aligned_slot_labels = aligner(data, multiple=True)
        for (key, entry), slot_label in zip(data.items(), aligned_slot_labels):
            entry['slot_labels'] = slot_label
        final_dir = f'../dataset/massive/slots/{CONFIG.lang}/{accent}'
        os.makedirs(final_dir, exist_ok=True)
        with open(f'{final_dir}/{split}.json', 'w+', encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
