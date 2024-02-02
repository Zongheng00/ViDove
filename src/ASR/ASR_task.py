# import threading
# import time

import openai
# from pytube import YouTube
# from os import getenv, getcwd
from pathlib import Path
# from enum import Enum, auto

import logging
import subprocess
from src.srt_util.srt import SrtScript
# from src.srt_util.srt2ass import srt2ass
# from time import time, strftime, gmtime, sleep
from src.translators.translation import get_translation, prompt_selector

import torch
import stable_whisper
# import shutil
# from datetime import datetime

def get_ASR(method, whisper_model, src_srt_path, source_lang, audio_path):

    if not Path.exists(src_srt_path):
        # extract script from audio
        logging.info("extract script from audio")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Module 1: ASR inference method: {method}")
        init_prompt = "Hello, welcome to my lecture." if source_lang == "EN" else ""
        if method == "api":
            with open(audio_path, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="srt", language=source_lang.lower(), prompt=init_prompt)
        elif method == "stable":
            model = stable_whisper.load_model(whisper_model, device)
            transcript = model.transcribe(str(audio_path), regroup=False,
                                                  initial_prompt=init_prompt)
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
        else:
            raise RuntimeError(f"unavaliable ASR inference method: {method}")   
            return None      