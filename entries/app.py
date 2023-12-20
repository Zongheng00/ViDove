import __init_lib_path
import gradio as gr
from src.task import Task
import logging
from yaml import Loader, Dumper, load, dump
import os
from pathlib import Path
from datetime import datetime
import shutil
from uuid import uuid4

launch_config = "./configs/local_launch.yaml"
task_config = './configs/task_config.yaml'

def init(output_type, src_lang, tgt_lang, domain):
    launch_cfg = load(open(launch_config), Loader=Loader)
    task_cfg = load(open(task_config), Loader=Loader)

    # overwrite config file
    task_cfg["source_lang"] = src_lang
    task_cfg["target_lang"] = tgt_lang
    task_cfg["field"] = domain

    if "Video File" in output_type:
        task_cfg["output_type"]["video"] = True
    else:
        task_cfg["output_type"]["video"] = False
    
    if "Bilingual" in output_type:
        task_cfg["output_type"]["bilingual"] = True
    else:
        task_cfg["output_type"]["bilingual"] = False
    
    if ".ass output" in output_type:
        task_cfg["output_type"]["subtitle"] = "ass"
    else:
        task_cfg["output_type"]["subtitle"] = "srt"

    # initialize dir
    local_dir = Path(launch_cfg['local_dump'])
    if not local_dir.exists():
        local_dir.mkdir(parents=False, exist_ok=False)

    # get task id
    task_id = str(uuid4())

    # create locak dir for the task
    task_dir = local_dir.joinpath(f"task_{task_id}")
    task_dir.mkdir(parents=False, exist_ok=False)
    task_dir.joinpath("results").mkdir(parents=False, exist_ok=False)

    # logging setting
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt, handlers=[
        logging.FileHandler(
            "{}/{}_{}.log".format(task_dir, f"task_{task_id}", datetime.now().strftime("%m%d%Y_%H%M%S")),
            'w', encoding='utf-8')])
    return task_id, task_dir, task_cfg

def process_input(video_file, youtube_link, src_lang, tgt_lang, domain, output_type):
    task_id, task_dir, task_cfg = init(output_type, src_lang, tgt_lang, domain)
    if youtube_link:
        task = Task.fromYoutubeLink(youtube_link, task_id, task_dir, task_cfg)
        task.run()
        return task.result
    elif video_file is not None:
        task = Task.fromVideoFile(video_file, task_id, task_dir, task_cfg)
        task.run()
        return task.result
    else:
        return None

demo = gr.Interface(fn=process_input,
    inputs=[
        gr.components.Video(label="Upload a video"),
        gr.components.Textbox(label="Or enter a YouTube URL"), 
        gr.components.Dropdown(choices=["EN", "ZH"], label="Select Source Language"),
        gr.components.Dropdown(choices=["ZH", "EN"], label="Select Target Language"),
        gr.components.Dropdown(choices=["General", "SC2"], label="Select Domain"),
        gr.CheckboxGroup(["Video File", "Bilingual", ".ass output"], label="Output Settings", info="What do you want?"),
    ],
    outputs=[
        gr.components.Video(label="Processed Video")
    ],
    title="ViDove: video translation toolkit demo",
    description="Upload a video or enter a YouTube URL."
    )

if __name__ == "__main__":
    demo.launch()