import zstandard
import os
import json
import pandas as pd
import logging
from datetime import datetime


# Set up logging
log = logging.getLogger("reddit_analysis")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


# Function to decode zstandard files
def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)
    
# Function to read lines from a zst file
def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line, file_handle.tell()
            buffer = lines[-1]
        reader.close()


# Function to process the raw file into a DataFrame
def load_reddit_data(file_path):
    log.info(f"Loading data from: {file_path}")
    file_size = os.stat(file_path).st_size
    data = []
    bad_lines = 0
    for line, _ in read_lines_zst(file_path):
        try:
            obj = json.loads(line)
            obj['created_datetime'] = datetime.utcfromtimestamp(int(obj['created_utc']))
            data.append(obj)
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
    log.info(f"Data loading complete with {len(data):,} rows and {bad_lines:,} bad lines.")
    return pd.DataFrame(data)