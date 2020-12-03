import seisbench

import requests
import ftplib
from tqdm import tqdm


def download_with_progress_bar(url, target):
    seisbench.logger.info(f"Downloading file from {url} to {target}")

    req = requests.get(url, stream=True, headers={"User-Agent": "SeisBench"})
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    pbar = tqdm(unit="B", total=total, desc="Downloading")

    with open(target, "wb") as f_target:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f_target.write(chunk)

    pbar.close()


def download_with_progress_bar_ftp(
    host, file, target, user="anonymous", passwd="", blocksize=8192
):
    with ftplib.FTP(host, user, passwd) as ftp:
        ftp.voidcmd("TYPE I")
        total = ftp.size(file)

        pbar = tqdm(unit="B", total=total, desc="Downloading")

        def callback(*args):
            pbar.update(blocksize)
            fout.write(*args)

        with open(target, "wb") as fout:
            ftp.retrbinary(f"RETR {file}", callback, blocksize=blocksize)

        pbar.close()
