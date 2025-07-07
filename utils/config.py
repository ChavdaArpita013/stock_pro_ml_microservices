import os
from dotenv import load_dotenv

load_dotenv()

#get value or fall back to default
MODEL_DIR = os.getenv("MODEL_DIR" , "model")

#make sure directory exists
os.makedirs(MODEL_DIR , exist_ok=True)