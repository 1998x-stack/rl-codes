# utils.py
from loguru import logger
import os

class Logger:
    def __init__(self, log_file="logs/training.log"):
        logger.add(log_file, rotation="1 MB")
    
    def log(self, message):
        logger.info(message)