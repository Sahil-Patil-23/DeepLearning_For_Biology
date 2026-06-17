FROM python:3-10.slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace/

RUN pip install -e .

CMD [ "python" , "Proteins_my_models/Training_in_Docker/train.py"]