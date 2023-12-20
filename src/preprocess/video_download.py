from pytube import YouTube
import logging

def download_youtube_to_local_file(youtube_url: str, local_dir_path: str = "./downloads") -> str:
    yt = YouTube(youtube_url)
    try:
        audio = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        # video = yt.streams.filter(file_extension='mp4').order_by('resolution').asc().first()
        if audio:
            saved_audio = audio.download(output_path=local_dir_path.join("/audio"))
            logging.info(f"Audio download successful: {saved_audio}")
            return saved_audio
        else:
            logging.error(f"Audio stream not found in {youtube_url}")
            raise f"Audio stream not found in {youtube_url}"
    except Exception as e:
        # print("Connection Error: ", end='')
        print(e)
        raise e

