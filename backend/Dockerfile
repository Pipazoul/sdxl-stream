# Use an official PyTorch image with CUDA 12 and Python 3.11 support
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install dependencies including FastAPI and any other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port on which FastAPI will run
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
