FROM tensorflow/tensorflow:latest-gpu-py3 as deep_extraction

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache

CMD python3 /home/emg-prediction/dataset/train_model.py