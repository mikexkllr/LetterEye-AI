def log_message(message):
    print(message)  # Simple logging to console, can be expanded to log to a file

def is_file_fully_copied(file_path, max_attempts=5, wait_time=1):
    import os
    import time

    attempts = 0
    while attempts < max_attempts:
        initial_size = os.path.getsize(file_path)
        time.sleep(wait_time)
        new_size = os.path.getsize(file_path)
        if initial_size == new_size:
            return True
        attempts += 1
    log_message(f"Failed to fully copy the file: {file_path}")
    return False