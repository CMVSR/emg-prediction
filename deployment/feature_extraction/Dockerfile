FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache

CMD python3 /home/emg-prediction/dataset/feature_extraction.py