# Project Setup Guide

## Prerequisites
Before starting, make sure you have installed:
- [Docker](https://www.docker.com/)
- [PostgreSQL 16](https://www.postgresql.org/)
- [Python 3.x](https://www.python.org/) (if not installed)

---
## 1️⃣ Set Up Docker & Database Image
📌 Step 1: Start Docker
Ensure that Docker is running on your system. If it is not started yet, launch Docker Desktop

📌 Step 2: Pull the pgvector Image
We will use ankane/pgvector for vector storage in PostgreSQL. Pull the latest image using:
```sh
docker pull ankane/pgvector:latest
```
## 2️⃣ Install Required Python Libraries
Run the following commands to install the necessary dependencies:
```sh
pip install langchain langchain_core langchain_text_splitters 
pip install langchain_google_genai
pip install langchain_community
```

## 3️⃣ Database Setup
📌 Step 1: Start PostgreSQL with Docker
```sh
docker compose up -d
```
📌 Step 2: Create database
```sh
docker exec -it postgres psql -U postgres -c "CREATE DATABASE new_legal;"
docker exec -it postgres psql -U postgres -c "CREATE DATABASE old_legal;"

```
📌 Step 3: Navigate to the Database Folder
```sh
cd database
```

📌 Step 4: Restore Data to the Databases
Add data to new_legal:
```sh
pg_restore -U postgres -d new_legal database/new_law.dump
```
Add data to old_legal:
```sh
pg_restore -U postgres -d old_legal database/old_law.dump
```
## 4️⃣ Run the Application
Navigate to the src folder and start the Streamlit application:
```sh
cd src
streamlit run streamlit_interface.py
```
🚀 The application is now running!
