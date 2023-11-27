FROM python:3.10.12

WORKDIR /code

COPY ./requirements-deploy.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src
COPY ./configs /code/configs
COPY ./main.py /code/
COPY ./.project-root /code/

# remove for prod, while committing
COPY ./logs/train/runs/2023-11-14_06-29-44/checkpoints/last.ckpt /code/model.ckpt
COPY ./.cache/checkpoints/dinov2_vitb14_reg4_pretrain.pth /code/.cache/checkpoints/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
