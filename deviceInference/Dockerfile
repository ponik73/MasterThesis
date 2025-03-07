# Step 1: Use the official Python image as a base
FROM python:3.11-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements.txt into the container
COPY requirements.txt .

# Step 4: Install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the FastAPI application into the container
COPY ./app /app

# Step 6: Expose the port that FastAPI will run on (default is 8000)
EXPOSE 8000

# Step 7: Set the command to run the app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]