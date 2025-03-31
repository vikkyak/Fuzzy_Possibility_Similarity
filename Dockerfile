FROM python:3.10

# Install dependencies
RUN pip install matplotlib scikit-learn networkx

# Set the working directory inside the container
WORKDIR /workspace

