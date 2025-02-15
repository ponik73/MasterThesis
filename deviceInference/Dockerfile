FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy only pyproject.toml first to leverage Docker cache
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the application files
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.11-slim

# WORKDIR /app

# FROM base as deps

# COPY pyproject.toml .

# RUN python -m pip install --no-cache-dir pip-tools && \
#     pip-compile -o requirements.txt pyproject.toml

# FROM base as dev

# COPY --from=deps /app/requirements.txt ./
# RUN python -m pip install --no-cache-dir -r requirements.txt

# # EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]