import seisbench

import requests
import ftplib
from tqdm import tqdm


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

    with open(target, "wb") as f_target:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if progress_bar:
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
