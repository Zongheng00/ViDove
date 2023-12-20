import logging
import subprocess
from argparse import ArgumentParser
from os import getenv
from pathlib import Path
from time import time, strftime, gmtime, sleep
from tqdm import tqdm
from datetime import datetime

import openai
import stable_whisper
import torch
import whisper
from pytube import YouTube

from src.srt_util.srt import SrtScript
from src.srt_util.srt2ass import srt2ass


def split_script(script_in, chunk_size=1000):
    script_split = script_in.split('\n\n')
    script_arr = []
    range_arr = []
    start = 1
    end = 0
    script = ""
    for sentence in script_split:
        if len(script) + len(sentence) + 1 <= chunk_size:
            script += sentence + '\n\n'
            end += 1
        else:
            range_arr.append((start, end))
            start = end + 1
            end += 1
            script_arr.append(script.strip())
            script = sentence + '\n\n'
    if script.strip():
        script_arr.append(script.strip())
        range_arr.append((start, len(script_split) - 1))

    assert len(script_arr) == len(range_arr)
    return script_arr, range_arr


def get_response(model_name, sentence):
    """
    Generates a translated response for a given sentence using a specified OpenAI model.

    :param model_name: The name of the OpenAI model to be used for translation, either "gpt-3.5-turbo" or "gpt-4".
    :param sentence: The English sentence related to StarCraft 2 videos that needs to be translated into Chinese.

    :return: The translated Chinese sentence, maintaining the original format, meaning, and number of lines.
    """

    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant that translates English to Chinese and have decent background in starcraft2."},
                # {"role": "system", "content": "Your translation has to keep the orginal format and be as accurate as possible."},
                # {"role": "system", "content": "Your translation needs to be consistent with the number of sentences in the original."},
                # {"role": "system", "content": "There is no need for you to add any comments or notes."},
                # {"role": "user", "content": 'Translate the following English text to Chinese: "{}"'.format(sentence)}

                {"role": "system",
                 "content": "你是一个翻译助理，你的任务是翻译星际争霸视频，你会被提供一个按行分割的英文段落，你需要在保证句意和行数的情况下输出翻译后的文本。"},
                {"role": "user", "content": sentence}
            ],
            temperature=0.15
        )

        return response['choices'][0]['message']['content'].strip()


def check_translation(sentence, translation):
    """
    check merge sentence issue from openai translation
    """
    sentence_count = sentence.count('\n\n') + 1
    translation_count = translation.count('\n\n') + 1

    if sentence_count != translation_count:
        # print("sentence length: ", len(sentence), sentence_count)
        # print("translation length: ",  len(translation), translation_count)
        return False
    else:
        return True


# Translate and save
def translate(srt, script_arr, range_arr, model_name, video_name, video_link, attempts_count=5):
    """
    Translates the given script array into another language using the chatgpt and writes to the SRT file.

    This function takes a script array, a range array, a model name, a video name, and a video link as input. It iterates
    through sentences and range in the script and range arrays. If the translation check fails for five times, the function
    will attempt to resolve merge sentence issues and split the sentence into smaller tokens for a better translation.

    :param srt: An instance of the Subtitle class representing the SRT file.
    :param script_arr: A list of strings representing the original script sentences to be translated.
    :param range_arr: A list of tuples representing the start and end positions of sentences in the script.
    :param model_name: The name of the translation model to be used.
    :param video_name: The name of the video.
    :param video_link: The link to the video.
    :param attempts_count: Number of attemps of failures for unmatched sentences.
    """
    logging.info("Start translating...")
    previous_length = 0
    for sentence, range_ in tqdm(zip(script_arr, range_arr)):
        # update the range based on previous length
        range_ = (range_[0] + previous_length, range_[1] + previous_length)

        # using chatgpt model
        print(f"now translating sentences {range_}")
        logging.info(f"now translating sentences {range_}, time: {datetime.now()}")
        flag = True
        while flag:
            flag = False
            try:
                translate = get_response(model_name, sentence)
                # detect merge sentence issue and try to solve for five times:
                while not check_translation(sentence, translate) and attempts_count > 0:
                    translate = get_response(model_name, sentence)
                    attempts_count -= 1

                # if failure still happen, split into smaller tokens
                if attempts_count == 0:
                    single_sentences = sentence.split("\n\n")
                    logging.info("merge sentence issue found for range", range_)
                    translate = ""
                    for i, single_sentence in enumerate(single_sentences):
                        if i == len(single_sentences) - 1:
                            translate += get_response(model_name, single_sentence)
                        else:
                            translate += get_response(model_name, single_sentence) + "\n\n"
                            # print(single_sentence, translate.split("\n\n")[-2])
                    logging.info("solved by individually translation!")

            except Exception as e:
                logging.debug("An error has occurred during translation:", e)
                print("An error has occurred during translation:", e)
                print("Retrying... the script will continue after 30 seconds.")
                sleep(30)
                flag = True

        srt.set_translation(translate, range_, model_name, video_name, video_link)


class Pigeon(object):
    def __init__(self):
        openai.api_key = getenv("OPENAI_API_KEY")
        self.v = False
        self.dir_download = None
        self.dir_result = None
        self.dir_log = None
        self.srt_path = None
        self.srt_only = False
        self.srt = None
        self.video_name = None
        self.video_path = None
        self.audio_path = None

        self.video_link = None
        self.video_file = None

        self.model = None

        self.parse()

        self.t_s = None
        self.t_e = None

    def parse(self):
        parser = ArgumentParser()
        parser.add_argument("--link", help="youtube video link here", type=str)
        parser.add_argument("--video_file", help="local video path", type=str)
        parser.add_argument("--video_name", help="video name, auto-filled if not provided")
        parser.add_argument("--audio_file", help="local audio path")
        parser.add_argument("--srt_file", help="srt file input path here", type=str)  # New argument
        parser.add_argument("--download", help="download path", default='./downloads')
        parser.add_argument("--output_dir", help="translate result path", default='./results')
        # default change to gpt-4
        parser.add_argument("--model_name", help="model name only support gpt-4 and gpt-3.5-turbo", default="gpt-4")
        parser.add_argument("--log_dir", help="log path", default='./logs')
        parser.add_argument("-only_srt", help="set script output to only .srt file", action='store_true')
        parser.add_argument("-v", help="auto encode script with video", action='store_true')
        args = parser.parse_args()

        self.v = args.v
        self.model = args.model_name
        self.srt_path = args.srt_file
        self.srt_only = args.only_srt

        # Set download path
        self.dir_download = Path(args.download)
        if not self.dir_download.exists():
            self.dir_download.mkdir(parents=False, exist_ok=False)
            self.dir_download.joinpath('audio').mkdir(parents=False, exist_ok=False)
            self.dir_download.joinpath('video').mkdir(parents=False, exist_ok=False)

        # Set result path
        self.dir_result = Path(args.output_dir)
        if not self.dir_result.exists():
            self.dir_result.mkdir(parents=False, exist_ok=False)

        # TODO: change if-else logic
        # Next, prepare video & audio files
        # Set video related
        if args.link is not None and (args.video_file is not None or args.audio_file is not None):
            raise ValueError("Please provide either video link or video/audio file path, not both.")
        if args.link is not None:
            self.video_link = args.link
            # Download audio from YouTube
            try:
                yt = YouTube(self.video_link)
                video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                if video:
                    video.download(str(self.dir_download.joinpath("video")))
                    print(f'Video download completed to {self.dir_download.joinpath("video")}!')
                else:
                    raise FileNotFoundError(f"Video stream not found for link {self.video_link}")
                audio = yt.streams.filter(only_audio=True, file_extension='mp4').first()
                if audio:
                    audio.download(str(self.dir_download.joinpath("audio")))
                    print(f'Audio download completed to {self.dir_download.joinpath("audio")}!')
                else:
                    raise FileNotFoundError(f"Audio stream not found for link {self.video_link}")
            except Exception as e:
                print("Connection Error: ", end='')
                print(e)
                raise ConnectionError
            self.video_path = self.dir_download.joinpath("video").joinpath(video.default_filename)
            self.audio_path = self.dir_download.joinpath("audio").joinpath(audio.default_filename)
            if args.video_name is not None:
                self.video_name = args.video_name
            else:
                self.video_name = Path(video.default_filename).stem
        else:
            if args.video_file is not None:
                self.video_path = args.video_file
                # Read from local video file
                self.video_path = args.video_file
                if args.video_name is not None:
                    self.video_name = args.video_name
                else:
                    self.video_name = Path(self.video_path).stem
                if args.audio_file is not None:
                    self.audio_path = args.audio_file
                else:
                    audio_path_out = self.dir_download.joinpath("audio").joinpath(f"{self.video_name}.mp3")
                    subprocess.run(['ffmpeg', '-i', self.video_path, '-f', 'mp3', '-ab', '192000', '-vn', audio_path_out])
                    self.audio_path = audio_path_out
            else:
                raise NotImplementedError("Currently audio file only not supported")

        if not self.dir_result.joinpath(self.video_name).exists():
            self.dir_result.joinpath(self.video_name).mkdir(parents=False, exist_ok=False)

        # Log setup
        self.dir_log = Path(args.log_dir)
        if not Path(args.log_dir).exists():
            self.dir_log.mkdir(parents=False, exist_ok=False)
        logging.basicConfig(level=logging.INFO, handlers=[
            logging.FileHandler(
                "{}/{}_{}.log".format(self.dir_log, self.video_name, datetime.now().strftime("%m%d%Y_%H%M%S")),
                'w', encoding='utf-8')])
        logging.info("---------------------Video Info---------------------")
        logging.info(
            f"Video name: {self.video_name}, translation model: {self.model}, video link: {self.video_link}")
        return

    def get_srt_class(self, whisper_model='tiny', method="stable"):
        # Instead of using the script_en variable directly, we'll use script_input
        if self.srt_path is not None:
            srt = SrtScript.parse_from_srt_file(self.srt_path)
        else:
            # using whisper to perform speech-to-text and save it in <video name>_en.txt under RESULT PATH.
            self.srt_path = Path(f"{self.dir_result}/{self.video_name}/{self.video_name}_en.srt")
            if not Path(self.srt_path).exists():
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # use OpenAI API for transcribe
                if method == "api":
                    with open(self.audio_path, "rb") as audio_file:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                # use local whisper model
                elif method == "basic":
                    # using base model in local machine (may use large model on our server)
                    model = whisper.load_model(whisper_model, device=device)
                    transcript = model.transcribe(self.audio_path)
                # use stable-whisper
                elif method == "stable":
                    # use cuda if available
                    model = stable_whisper.load_model(whisper_model, device=device)
                    transcript = model.transcribe(str(self.audio_path), regroup=False,
                                                  initial_prompt="Hello, welcome to my lecture. Are you good my friend?")
                    (
                        transcript
                        .split_by_punctuation(['.', '。', '?'])
                        .merge_by_gap(.15, max_words=3)
                        .merge_by_punctuation([' '])
                        .split_by_punctuation(['.', '。', '?'])
                    )
                    transcript = transcript.to_dict()
                else:
                    raise ValueError("invalid speech to text method")

                srt = SrtScript(transcript['segments'])  # read segments to SRT class
            else:
                srt = SrtScript.parse_from_srt_file(self.srt_path)
        self.srt = srt
        return

    def preprocess(self):
        self.t_s = time()
        self.get_srt_class()
        # SRT class preprocess
        logging.info("--------------------Start Preprocessing SRT class--------------------")
        self.srt.write_srt_file_src(self.srt_path)
        self.srt.form_whole_sentence()
        # self.srt.spell_check_term()
        self.srt.correct_with_force_term()
        processed_srt_file_en = str(Path(self.srt_path).with_suffix('')) + '_processed.srt'
        self.srt.write_srt_file_src(processed_srt_file_en)
        script_input = self.srt.get_source_only()

        # write ass
        if not self.srt_only:
            logging.info("write English .srt file to .ass")
            assSub_en = srt2ass(processed_srt_file_en, "default", "No", "Modest")
            logging.info('ASS subtitle saved as: ' + assSub_en)
        return script_input

    def start_translation(self, script_input):
        script_arr, range_arr = split_script(script_input)
        logging.info("---------------------Start Translation--------------------")
        translate(self.srt, script_arr, range_arr, self.model, self.video_name, self.video_link)

    def postprocess(self):
        # SRT post-processing
        logging.info("---------------------Start Post-processing SRT class---------------------")
        self.srt.check_len_and_split()
        self.srt.remove_trans_punctuation()

        base_path = Path(self.dir_result).joinpath(self.video_name).joinpath(self.video_name)

        self.srt.write_srt_file_translate(f"{base_path}_zh.srt")
        self.srt.write_srt_file_bilingual(f"{base_path}_bi.srt")

        # write ass
        if not self.srt_only:
            logging.info("write Chinese .srt file to .ass")
            assSub_zh = srt2ass(f"{base_path}_zh.srt", "default", "No", "Modest")
            logging.info('ASS subtitle saved as: ' + assSub_zh)

        # encode to .mp4 video file
        if self.v:
            logging.info("encoding video file")
            if self.srt_only:
                subprocess.run(
                    f'ffmpeg -i {self.video_path} -vf "subtitles={base_path}_zh.srt" {base_path}.mp4')
            else:
                subprocess.run(
                    f'ffmpeg -i {self.video_path} -vf "subtitles={base_path}_zh.ass" {base_path}.mp4')

        self.t_e = time()
        logging.info(
            "Pipeline finished, time duration:{}".format(strftime("%H:%M:%S", gmtime(self.t_e - self.t_s))))

    def run(self):
        script_input = self.preprocess()
        self.start_translation(script_input)
        self.postprocess()
