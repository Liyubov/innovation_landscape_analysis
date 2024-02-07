# Use an official Python runtime as a parent image
FROM python:3.9-slim

RUN mkdir -p /app/model/

# Set the working directory to /app
WORKDIR /app

COPY . /app

# Copy the current directory contents into the container at /app
COPY model /app/model

RUN apt-get update
RUN apt-get install -y curl gnupg2
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8501


# Run main.py when the container launches
CMD ["streamlit", "run", "main.py"]