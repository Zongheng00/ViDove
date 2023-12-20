import pathlib
import os
import subprocess


def extract_audio(local_video_path: str, save_dir_path: str = "./downloads/audio") -> str:
    if os.name == 'nt':
        NotImplementedError("Filename extraction on Windows not yet implemented")

    out_file_name = os.path.basename(local_video_path)
    audio_path_out = save_dir_path.join("/").join(out_file_name)
    subprocess.run(['ffmpeg', '-i', local_video_path, '-f', 'mp3', '-ab', '192000', '-vn', audio_path_out])
    return audio_path_out