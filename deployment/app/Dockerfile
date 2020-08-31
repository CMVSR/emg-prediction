FROM tensorflow/tensorflow:latest-gpu-py3 as app

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache

CMD python3 /home/emg-prediction/app.py
