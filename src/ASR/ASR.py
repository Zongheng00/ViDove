import openai
from pathlib import Path

import logging
import torch
import stable_whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def get_transcript(method, whisper_model, src_srt_path, source_lang, audio_path):

    istrans = False # is trans flag 

    if not Path.exists(src_srt_path):
        # extract script from audio
        logging.info("extract script from audio")
        logging.info(f"Module 1: ASR inference method: {method}")
        init_prompt = "Hello, welcome to my lecture." if source_lang == "EN" else ""

        # process the audio by method
        # TODO: method "api" should be changed to "whisper1" afterwards
        if method == "api": 
            transcript = get_transcript_whisper1(audio_path, source_lang, init_prompt)
            istrans = True
        elif method == "whisper-large-v3":
            transcript = get_transcript_whisper_large_v3(audio_path)
            istrans = True
        elif method == "stable":
            transcript = get_transcript_stable(audio_path, whisper_model, init_prompt)
            istrans = True
        else:
            raise RuntimeError(f"unavaliable ASR inference method: {method}")   
    
    # return transcript or None
    if (istrans == True):
        return transcript    
    else: 
        return None
        
def get_transcript_whisper1(audio_path, source_lang, init_prompt):
    with open(audio_path, 'rb') as audio_file:
        transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="srt", language=source_lang.lower(), prompt=init_prompt)
    return transcript
    
def get_transcript_stable(audio_path, whisper_model, init_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = stable_whisper.load_model(whisper_model, device)
    transcript = model.transcribe(str(audio_path), regroup=False, initial_prompt=init_prompt)
    (
        transcript
        .split_by_punctuation(['.', '。', '?'])
        .merge_by_gap(.15, max_words=3)
        .merge_by_punctuation([' '])
        .split_by_punctuation(['.', '。', '?'])
    )
    transcript = transcript.to_dict()
    transcript = transcript['segments']
    # after get the transcript, release the gpu resource
    torch.cuda.empty_cache()

    return transcript

def get_transcript_whisper_large_v3(audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    )

    transcript_whisper_v3 = pipe(str(audio_path))

    # convert format
    transcript = []
    for i in transcript_whisper_v3['chunks']:
        transcript.append({'start': i['timestamp'][0], 'end': i['timestamp'][1], 'text':i['text']})

    return transcript