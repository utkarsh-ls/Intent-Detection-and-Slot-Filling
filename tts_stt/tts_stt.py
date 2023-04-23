# %%
import os
import argparse

parser = argparse.ArgumentParser(description='Get configurations to train')
parser.add_argument('--accent', default='orig', type=str)
parser.add_argument('--lang', default='en', type=str)
parser.add_argument('--type', default='train', type=str)
CONFIG = parser.parse_args()
# %%
from datasets import load_dataset

if CONFIG.lang == 'en':
    dataset = load_dataset("AmazonScience/massive", "en-US")
else:
    dataset = load_dataset("AmazonScience/massive", "hi-IN")

# %%
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

if CONFIG.accent == 'orig':
    import json

    for data_type in dataset.keys():
        data = dataset[data_type]
        orig_data = {}
        for i in range(len(data)):
            orig_data[i] = data[i]
        
        dir = '../dataset/massive/'

        dir += f'{CONFIG.lang}/{CONFIG.accent}/'

        os.makedirs(dir, exist_ok=True)

        path = dir+f'{data_type}.json'
        file = open(f'{path}', "w+", encoding="utf-8")
        json.dump(orig_data, file, ensure_ascii=False)
        file.close()
    
    exit(0)

# %%
import whisper
import torch

asr_model = whisper.load_model("small")
device = torch.device('cuda')
asr_model = asr_model.to(device)


# %%
if CONFIG.lang == 'en':
    language = 'en'
    model_id = 'v3_en_indic'
else:
    language = 'indic'
    model_id = 'v3_indic'

accent = CONFIG.accent
# DATA TYPES: train, validation, test
data_type = CONFIG.type

tts_model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
tts_model.to(device)  # gpu or cpu

# %%
def tts(text, speaker='hindi_female'):
    sample_rate = 48000
    audio = tts_model.save_wav(text=text,
                            speaker=speaker,
                            sample_rate=sample_rate)
    return audio

# %%
def asr(audio):
    result = asr_model.transcribe(audio)
    return result["text"]

# %%
# pip install -U playsound

# %%
# from playsound import playsound

# %%
# data_type = "train"

# from tqdm import tqdm
# data = dataset[data_type]

# asr_data = {}
# speech = {}
# for i in tqdm(range(3)):
#     asr_data[i] = data[i]
#     speech[i] = tts(data[i]["utt"], speaker='hindi_female')
#     asr_data[i]["utt"] = asr(speech[i])
#     print(data[i])
#     print(asr_data[i]["utt"])

# %%
# pip install -U pydub

# %%
# from pydub import AudioSegment
# from pydub.playback import play
 
# # Import an audio file
# # Format parameter only
# # for readability
# wav_file = AudioSegment.from_file(file = speech[0],
#                                   format = "wav")
 
# # Play the audio file
# play(wav_file)

# %%
# print(asr_data)
# print(test[:3]['utt'])

# %%
from tqdm import tqdm
data = dataset[data_type]

asr_data = {}
speech = {}
for i in tqdm(range(len(data))):
    asr_data[i] = data[i]
    try:
        speech[i] = tts(data[i]["utt"], speaker=accent)
        asr_data[i]["utt"] = asr(speech[i])
    except:
        continue

# %% [markdown]
# Small 15to1.6 it/s

# %%
import json

dir = '../dataset/massive/'

dir += f'{CONFIG.lang}/{accent}/'

os.makedirs(dir, exist_ok=True)

path = dir+f'{data_type}.json'
file = open(f'{path}', "w+", encoding="utf-8")
json.dump(asr_data, file, ensure_ascii=False)
file.close()

# %%
os.listdir(dir)

# %%
orig_data = {}
for i in range(len(data)):
    orig_data[i] = data[i]

# file = open("/content/drive/MyDrive/Colab Notebooks/test_small_eng.json", "w+")
# json.dump(orig_data, file)
# file.close()

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# import json

# file = open("/content/drive/MyDrive/Colab Notebooks/test_small_tamil_f.json")
# asr_data = json.load(file)
# file.close()

# file = open("/content/drive/MyDrive/Colab Notebooks/test_small_tamil.json")
# orig_data = json.load(file)
# file.close()

# %%
txt_orig = [orig_data[x]['utt'] for x in orig_data]
txt_new = [asr_data[x]['utt'] for x in asr_data]

# %%
txt_orig_emb = model.encode(txt_orig)
txt_new_emb = model.encode(txt_new)

# %%
# Cosine similarity

sim = [round(cosine_similarity([t_o], [t_n])[0][0], 3) for t_o, t_n in zip(txt_orig_emb, txt_new_emb)]
print(sim)

# %%
import numpy as np

# %matplotlib inline
# plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

# # Plot Histogram on x
# plt.hist(sim, bins=300)
# plt.gca().set(title='Similarity (0-1)', ylabel='Frequency', xlim=(0.9, 1))

print(np.mean(np.array(sim)))
print(np.std(np.array(sim)))

# %%
# Word error rate
from jiwer import wer

err = wer(txt_orig, txt_new)
print(err)

# English Accents (15)
## tamil_female
## bengali_female
## malayalam_male
## manipuri_female
## assamese_female
## gujarati_male
## telugu_male
## kannada_male
## hindi_female
## rajasthani_female
## kannada_female
## bengali_male
## tamil_male
## gujarati_female
## assamese_male

# Hindi Accents (17)
## bengali_female
## bengali_male
## gujarati_female
## gujarati_male
## hindi_female
## hindi_male
## kannada_female
## kannada_male
## malayalam_female
## malayalam_male
## manipuri_female
## rajasthani_female
## rajasthani_male
## tamil_female
## tamil_male
## telugu_female
## telugu_male

