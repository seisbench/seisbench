import seisbench

import requests
import ftplib
from tqdm import tqdm
from pathlib import Path
import time
import os


def download_http(url, target, progress_bar=True, desc="Downloading"):
    seisbench.logger.info(f"Downloading file from {url} to {target}")

    req = requests.get(url, stream=True, headers={"User-Agent": "SeisBench"})

    if req.status_code != 200:
        raise ValueError(
            f"Invalid URL. Request returned status code {req.status_code}."
        )

    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if progress_bar:
        pbar = tqdm(unit="B", total=total, desc=desc)
    else:
        pbar = None

    target = Path(target)

    with open(target, "wb") as f_target:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if pbar is not None:
                    pbar.update(len(chunk))
                f_target.write(chunk)

    if progress_bar:
        pbar.close()


def download_ftp(
    host,
    file,
    target,
    user="anonymous",
    passwd="",
    blocksize=8192,
    progress_bar=True,
    desc="Downloading",
):
    with ftplib.FTP(host, user, passwd) as ftp:
        ftp.voidcmd("TYPE I")
        total = ftp.size(file)

        if progress_bar:
            pbar = tqdm(unit="B", total=total, desc=desc)

        def callback(chunk):
            if progress_bar:
                pbar.update(len(chunk))
            fout.write(chunk)

        with open(target, "wb") as fout:
            ftp.retrbinary(f"RETR {file}", callback, blocksize=blocksize)

        if progress_bar:
            pbar.close()


def callback_if_uncached(
    files, callback, force=False, wait_for_file=False, test_interval=60
):
    """
    Checks if all files exists and executes the callback otherwise.
    Please note that the callback is executed if *at least one* file is not cached.
    If one of the files does not exists, but file.partial does, the behaviour depends on force and wait_for_file.

    WARNING: While making concurrent callbacks unlikely, they can still happen, if the function is called twice in short time,
    i.e., the second starts before the first created a .partial file.
    :param files: A list of files to check.
    :param callback: A callback, taking one parameter, a list of target file names.
    Will be called if a file is missing.
    The callback will be given the same parameter as provided in files, just with files renamed to file.partial.
    The function will move the files afterwards, but will ignore empty files.
    :param force: If true, and not all files exist, ignore and remove all partial files and execute callback.
    Only use this parameter if no other instance of callback_if_uncached is currently requesting the same file.
    :param wait_for_file: If true, not all files exist, but partial files exist, sleep until files exists or no partial files exist.
    :param test_interval: Sleep interval for wait_for_file.
    :return: None
    """
    if not isinstance(files, (list, tuple)):
        files = [files]
        squeeze = True
    else:
        squeeze = False

    files = [Path(file) for file in files]
    partial_files = [file.parent / (file.name + ".partial") for file in files]

    def exist():
        return all(file.is_file() for file in files)

    def is_partial():
        return any(file.is_file() for file in partial_files)

    while not exist() and is_partial() and not force:
        if wait_for_file:
            seisbench.logger.warning(
                f"Found partial instance. Rechecking in {test_interval} seconds."
            )
            time.sleep(test_interval)
        else:
            raise ValueError(
                f"Found partial instance. "
                f"This suggests that either the download is currently in progress or a download failed. "
                f"To redownload the file, call the dataset with force=True. "
                f"To wait for another download to finish, use wait_for_file=True."
            )

    if not exist() and force:
        for file in files:
            partial_file = file.parent / (file.name + ".partial")
            if partial_file.is_file():
                os.remove(partial_file)

    if exist():
        return
    else:
        # Open and close each partial file once, to ensure they exist early and make race conditions unlikely.
        for file in partial_files:
            file.parent.mkdir(parents=True, exist_ok=True)
            open(file, "a").close()

        try:
            if squeeze:
                callback(partial_files[0])
            else:
                callback(partial_files)
        finally:
            for partial_file in partial_files:
                if partial_file.stat().st_size == 0:
                    os.remove(partial_file)

        seisbench.logger.info("Moving partial files to target")
        for partial_file, file in zip(partial_files, files):
            if partial_file.is_file():
                partial_file.rename(file)
