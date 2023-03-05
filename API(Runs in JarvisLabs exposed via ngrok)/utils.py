import os
import torch
from glob import glob

from pytube import YouTube
import moviepy.editor as mp

import torch
import librosa
import shutil
import soundfile as sf
from tqdm import tqdm


from transformers import Wav2Vec2Processor, HubertForCTC,\
                        BartForConditionalGeneration, BartTokenizer,\
                        AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("STARTING MODELS LOAD")

transcript_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
transcript_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

print("TRANSCRIPT MODELS LOADED")

paraphrase_tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
paraphrase_model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase').to(device)

print("PARAPHRASING MODELS LOADED")

summary_tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
summary_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum").to(device)

print("SUMMARY MODELS LOADED")


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

    buffer = 30 * sr
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
        data, sampling_rate = librosa.load('chunks/' + path, sr=16000)
        inputs = transcript_processor(data, 
                                      sampling_rate = sampling_rate, 
                                      return_tensors="pt")
        
        inputs = {k:v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            logits = transcript_model(**inputs).logits

        ids = torch.argmax(logits, dim=-1)
        transcription = transcript_processor.batch_decode(ids)[0]
        text += transcription + " "
    
    shutil.rmtree('chunks')
    return name, text

def paraphrase_transcript(url = None, filepath = None):
    if url is not None:
        name, transcript = fetch_transcripts(url = url)
    
    elif filepath is not None:
        name, transcript = fetch_transcripts(filepath = filepath)

    batch = paraphrase_tokenizer(transcript, return_tensors='pt')
    generated_ids = paraphrase_model.generate(batch['input_ids'].to(device))
    
    return name, paraphrase_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def summarize(url = None, filepath = None):
    if url is not None:
        name, text = fetch_transcripts(url = url)
    
    elif filepath is not None:
        name, text = fetch_transcripts(filepath = filepath)
    
    length = len(text.split())
    
    print('-----------------------------------EXTRACTING SUMMARY----------------------------------------')

    tokens = summary_tokenizer(text.lower(), truncation=True, padding="longest", return_tensors="pt")
    
    tokens = {
        'input_ids': tokens['input_ids'].to(device), 
        'attention_mask': tokens['attention_mask'].to(device)
    }

    summary = summary_model.generate(**tokens)
    return name.split('/')[-1], summary_tokenizer.decode(summary[0], 
                                  skip_special_tokens = True,
                                  temperature = 0.7,
                                  no_repeat_ngram_size=2)