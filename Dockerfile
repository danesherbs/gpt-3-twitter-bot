FROM pytorch/pytorch:latest
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY main.py models.py my_gpt.pt /
CMD [ "python", "/main.py" ]
