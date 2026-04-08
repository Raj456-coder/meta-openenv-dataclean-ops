FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    gradio \
    pydantic \
    numpy \
    pandas \
    scipy \
    rich \
    openai

# Copy all files
COPY . .

EXPOSE 7860

# Run Gradio app
CMD ["python", "app.py"]