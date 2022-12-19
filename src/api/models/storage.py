import hashlib
import os


def get_name_and_ext(filename: str):
    index = filename.rfind(".")
    if index < 0:  # no extension in filename
        return filename, ""

    return filename[:index], filename[index:]  # ext with leading dot (e.q. ".txt")


def get_hashed_filename(filename, time):
    _, ext = get_name_and_ext(filename)
    hashed_filename_with_ext = hashlib.md5(f"{filename}_{time}".encode(encoding="utf-8")).hexdigest() + ext
    return hashed_filename_with_ext


def get_file_path_with_folder_name(instance, filename, folder_name, calculate_by_updated_time=False):
    time = instance.created_at.strftime("%Y/%m/%d")
    if calculate_by_updated_time:
        time = instance.updated_at.strftime("%Y/%m/%d")

    hashed_filename_with_ext = get_hashed_filename(filename, time)
    return os.path.join(f"{folder_name}/{time}/", hashed_filename_with_ext)
