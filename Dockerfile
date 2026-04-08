FROM python:3.11-slim

WORKDIR /app

# Install all dependencies
RUN pip install --no-cache-dir \
    pydantic>=2.0.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    rich>=13.0.0 \
    openai>=1.0.0 \
    gradio>=4.0.0

# Copy all application files
COPY models.py env.py tasks.py baseline.py server.py __init__.py openenv.yaml ./
COPY inference.py app.py requirements.txt ./

EXPOSE 7860

# Run Gradio app for HuggingFace Spaces
CMD ["python", "app.py"]