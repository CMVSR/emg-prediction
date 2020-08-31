FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache

# VSCODE DEBUG
# CMD echo Waiting for debugger to attach... && \
#     python3 -m ptvsd --host 0.0.0.0 --port 5678 --wait /home/emg-prediction/app.py

CMD python3 /home/emg-prediction/app.py