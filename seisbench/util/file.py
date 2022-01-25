import seisbench

import requests
import ftplib
from tqdm import tqdm
from pathlib import Path
import time
import os
from lxml import etree


def download_http(
    url, target, progress_bar=True, desc="Downloading", precheck_timeout=3
):
    """
    Downloads file from http/https source. Raises a ValueError for non-200 status codes.

    :param url: Target url
    :type url: str
    :param target: Path to save to
    :type target: Path or str
    :param progress_bar: If true, shows a progress bar for the download
    :type progress_bar: bool
    :param desc: Description for the progress bar
    :type desc: str
    :param precheck_timeout: Timeout passed to :py:func:`precheck_url`
    :type precheck_timeout: int
    """
    seisbench.logger.info(f"Downloading file from {url} to {target}")

    precheck_url(url, timeout=precheck_timeout)

    req = requests.get(url, stream=True, headers={"User-Agent": "SeisBench"})

    if req.status_code != 200:
        raise ValueError(
            f"Invalid URL. Request returned status code {req.status_code}."
        )

    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if progress_bar:
        pbar = tqdm(
            unit="B", total=total, desc=desc, unit_scale=True, unit_divisor=1024
        )
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


def precheck_url(url, timeout):
    """
    Checks whether the url is reachable and give a 200 or 300 HTTP response code.
    If a timeout occurs or a >=400 response code is returned, the precheck issues a warning.

    :param url: URL to check
    :param timeout: Timeout in seconds
    """
    if timeout <= 0:
        # Skip check
        return

    error_description = None

    try:
        req = requests.head(url, timeout=timeout, headers={"User-Agent": "SeisBench"})

        if req.status_code >= 400:  # Assumes all 200 and 300 codes are acceptable
            error_description = f"status code {req.status_code}"

    except requests.Timeout:
        error_description = "a timeout"

    except requests.ConnectionError:
        error_description = "a connection error"

    if error_description is not None:
        seisbench.logger.warning(
            f"The download precheck failed with {error_description}. "
            f"This is not an error itself, but might indicate a subsequent error. "
            f"If you encounter an error, this might be caused by the firewall setup of your "
            f"network. "
            f"Please check https://github.com/seisbench/seisbench#known-issues for details."
        )


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
    """
    Downloads file from ftp source.

    :param host: Host URL
    :type host: str
    :param file: File path on the FTP server
    :type file: str
    :param target: Path to save to
    :type target: Path or str
    :param user: Username for login
    :type user: str
    :param passwd: Password for login
    :type passwd: str
    :param blocksize: Size of download blocks in bytes
    :type blocksize: int
    :param progress_bar: If true, shows a progress bar for the download
    :type progress_bar: bool
    :param desc: Description for the progress bar
    :type desc: str
    """
    with ftplib.FTP(host, user, passwd) as ftp:
        ftp.voidcmd("TYPE I")
        total = ftp.size(file)

        if progress_bar:
            pbar = tqdm(
                unit="B", total=total, desc=desc, unit_scale=True, unit_divisor=1024
            )

        def callback(chunk):
            if progress_bar:
                pbar.update(len(chunk))
            fout.write(chunk)

        with open(target, "wb") as fout:
            ftp.retrbinary(f"RETR {file}", callback, blocksize=blocksize)

        if progress_bar:
            pbar.close()


def ls_webdav(url, precheck_timeout=3):
    """
    Lists the files in a WebDAV directory

    :param url: URL of the directory to list
    :type url: str
    :param precheck_timeout: Timeout passed to :py:func:`precheck_url`
    :type precheck_timeout: int
    :return: List of files
    """
    precheck_url(url, timeout=precheck_timeout)

    xml_request = b'<?xml version="1.0"?><a:propfind xmlns:a="DAV:"><a:prop><a:resourcetype/></a:prop></a:propfind>'

    req = requests.Request("PROPFIND", url, headers={"Depth": "1"}, data=xml_request)
    prep = req.prepare()

    with requests.Session() as sess:
        ret = sess.send(prep)

    if not 200 <= ret.status_code < 300:
        raise ValueError(
            f"Invalid URL. Request returned status code {ret.status_code}."
        )

    files = ["."]  # The first part of the response is always the root path
    tree = etree.fromstring(ret.content)
    for elem in tree.xpath(".//d:href", namespaces={"d": "DAV:"})[1:]:
        # Extract basepath from path
        p = elem.text[:-1].rfind("/") + 1
        files.append(elem.text[p:])

    return files


def callback_if_uncached(
    files, callback, force=False, wait_for_file=False, test_interval=60
):
    """
    Checks if all files exists and executes the callback otherwise.
    Please note that the callback is executed if *at least one* file is not cached.
    If one of the files does not exists, but file.partial does, the behaviour depends on force and wait_for_file.

    .. warning::
        While making concurrent callbacks unlikely, they can still happen, if the function is called twice in short
        time, i.e., the second starts before the first created a .partial file.

    :param files: A list of files or single file to check.
    :type files: list[union[Path, str]], Path, str
    :param callback: A callback, taking one parameter, a list of target file names. Will be called if a file is missing.
                     The callback will be given the same parameter as provided in files, just with files renamed
                     to file.partial. The function will move the files afterwards, but will ignore empty files.
    :type callback: callable
    :param force: If true, and not all files exist, ignore and remove all partial files and execute callback. Only use
                  this parameter if no other instance of callback_if_uncached is currently requesting the same file.
    :type force: bool
    :param wait_for_file: If true, not all files exist, but partial files exist, sleep until files exists or no partial
                          files exist.
    :type wait_for_file: bool
    :param test_interval: Sleep interval for wait_for_file.
    :type test_interval: float
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
