import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSeq2SeqLM

from glob import glob
from pytube import YouTube
import moviepy.editor as mp

import os
import librosa
import soundfile as sf

import uuid
import boto3
import shutil
import numpy as np
from tqdm.notebook import tqdm

ACCESS_ID = 'AKIA3VWL7P6ZRDE2QWHU'
ACCESS_KEY = 'gOxxfmYkXdIsaMjfR6GRaclkzNJtiut4X2PjyCLj'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('-----------------------------------EXTRACTING ASR-----------------------------------------------')

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

print('-----------------------------------EXTRACTED ASR-----------------------------------------------')
print('-----------------------------------EXTRACTING SUMM-----------------------------------------------')

tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum").to(device)

print('-----------------------------------EXTRACTED SUMM-----------------------------------------------')

def fetch_audio(url = None, filepath = None):
    print('-----------------------------------EXTRACTING AUDIO-----------------------------------------------')
    if url is not None:
        yt = YouTube(url).streams.filter(progressive=True, file_extension='mp4')
        filepath = yt.first().download()
    
        clip = mp.VideoFileClip(filepath)
        clip.audio.write_audiofile('converted.wav')
    
    elif filepath is not None:
        clip = mp.VideoFileClip(filepath)
        clip.audio.write_audiofile('converted.wav')

    os.remove(filepath)
    return filepath.split('.')[0]

def fetch_chunks(url = None, filepath = None):
    if url is not None:
        name = fetch_audio(url = url)
    
    elif filepath is not None:
        name = fetch_audio(filepath = filepath)
    
    print('-----------------------------------SPLITTING AUDIO------------------------------------------------')
    speech, sr = librosa.load('converted.wav', sr=16000)

    buffer = 20 * sr
    samples_total = len(speech)
    samples_wrote = 0
    counter = 1

    if not os.path.exists('chunks'):
        os.makedirs('chunks')
        
    pbar = tqdm(total=int(samples_total//buffer))
    while samples_wrote < samples_total:
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        block = speech[samples_wrote : (samples_wrote + buffer)]
        out_filename = f'chunks/split_{counter}.wav'

        sf.write(out_filename, block, sr)
        counter += 1

        samples_wrote += buffer
        pbar.update(1)
    os.remove('converted.wav')
    return name

def fetch_transcripts(url = None, filepath = None):
    if url is not None:
        name = fetch_chunks(url = url)
    
    elif filepath is not None:
        name = fetch_chunks(filepath = filepath)
    
    print('-----------------------------------EXTRACTING TRANSCRIPTS----------------------------------------')

    text = ""
    for path in tqdm(os.listdir('chunks')):
        speech, sr = librosa.load('chunks/' + path, sr=16000)
        input_values = processor(speech, return_tensors="pt").input_values

        logits = asr(input_values.to(device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text += processor.batch_decode(predicted_ids)[0] + " "
    
    shutil.rmtree('chunks')
    return name, text

def summarize(url = None, filepath = None):
    if url is not None:
        name, text = fetch_transcripts(url = url)
    
    elif filepath is not None:
        name, text = fetch_transcripts(filepath = filepath)
    
    length = len(text.split())
    print(text)
    print('-----------------------------------EXTRACTING SUMMARY----------------------------------------')

    tokens = tokenizer(text.lower(), truncation=True, padding="longest", return_tensors="pt")
    
    tokens = {
        'input_ids': tokens['input_ids'].to(device), 
        'attention_mask': tokens['attention_mask'].to(device)
    }

    summary = model.generate(**tokens)
    return name.split('/')[-1], tokenizer.decode(summary[0], 
                                  skip_special_tokens = True,
                                  temperature = 0.7,
                                  no_repeat_ngram_size=2)