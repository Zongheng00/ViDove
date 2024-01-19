# set base image (host OS)
FROM python:3.10.13-bullseye

# set the environment variable, you should put your own key here
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ARG GRADIO_SERVER_PORT=8301
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
ARG GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME}

# set the working directory in the container
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --upgrade pip

# install fonts
RUN mkdir -p /usr/local/share/fonts/truetype
COPY ./fonts/*.otf /usr/local/share/fonts/truetype/
COPY ./fonts/*.ttc /usr/local/share/fonts/truetype/
RUN fc-cache -f -v
RUN rm ./fonts/*.otf
RUN rm ./fonts/*.ttc

# install ffmpeg
RUN apt-get -y update \ 
    && apt-get -y upgrade \ 
    && apt-get install -y --no-install-recommends ffmpeg

RUN pip install --no-cache-dir -r requirements.txt

# expose the port
EXPOSE ${GRADIO_SERVER_PORT}

CMD ["python", "entries/app.py"]