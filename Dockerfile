FROM python:3.12-slim

WORKDIR /usr/project
COPY . .
RUN pip install -r ./requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "src/app.py"]