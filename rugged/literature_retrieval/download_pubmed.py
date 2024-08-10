from ftplib import FTP
import re
import os
import subprocess
import time
import urllib.request


def separate_files_by_year(files):
    """
    Returns a mapping of year to file names for PubMed documents.
    """
    year_to_files = {}
    for file in files:
        # Match files that end with .xml.gz and capture the two-digit year
        match = re.match(r'pubmed(\d{2})n\d+\.xml\.gz$', file)
        if match:
            year = '20' + match.group(1)  # Convert to four-digit year
            if year not in year_to_files:
                year_to_files[year] = []
            year_to_files[year].append(file)
    return year_to_files


def filter_files_by_years(year_to_files, years_to_include):
    """
    Filters files to include only those matching specified years.
    """
    files_to_download = []
    for year, files in year_to_files.items():
        if year in years_to_include:
            files_to_download.extend(files)
    return files_to_download


def download_file(url, destination):
    """
    Download a file from a URL to a specified destination.
    """
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def check_md5(file, mac=False, linux=True):
    """
    Verify one file via md5. Assumes Linux or MacOS
    """

    if os.path.isfile(file) and os.path.isfile(file + ".md5"):

        # Get md5 output from the zipped .xml
        if linux == True:
            stdout = subprocess.check_output('md5sum '+file, shell=True).decode('utf-8')
        elif mac == True:
            stdout = subprocess.check_output('md5 '+file, shell=True).decode('utf-8')
        md5_calculated = re.search('[0-9a-f]{32}', stdout).group(0)

        # Get md5 output from the corresponding .md5 file
        md5 = re.search('[0-9a-f]{32}', open(file + '.md5', 'r').readline()).group(0)

        # Check if the md5 are equal
        return md5 == md5_calculated

def extract(file, logfile):
    """
    Extract data from the zipped downloaded PubMed data.
    """
    try:
        subprocess.check_call(['gunzip', '-fq', file])
        logfile.write(f"Extraction successful for {file}\n")
    except subprocess.CalledProcessError as e:
        logfile.write(f"Error: Extraction failed for {file}. Return code: {e.returncode}\n")
        raise


def download_document(f, checksum, base_url, retry_limit=3):
    """
    Downloads and verifies a .xml.gz file and its .xml.gz.md5 checksum.
    Retries download and verification up to 'retry_limit' times if an error occurs.
    """
    download_attempts = 0
    logfile = open("download_log.txt", "a")

    while download_attempts < retry_limit:
        # Attempt to download both the file and its checksum
        file_downloaded = download_file(base_url + f, f)
        checksum_downloaded = download_file(base_url + checksum, checksum)

        if not file_downloaded or not checksum_downloaded:
            download_attempts += 1
            time.sleep(2)  # Wait a bit before retrying
            continue  # Skip the rest of the loop and try again

        # Verify the checksum only if both files are successfully downloaded
        if check_md5(f):
            print(f"Download and verification successful for {f}")
            # Extract the .xml.gz file after successful verification
            try:
                extract(f, logfile)
                break  # Successfully downloaded, extracted, and verified; exit the loop
            except Exception as exc:
                print(f"Extraction failed for {f}: {exc}")
                # If extraction fails, consider it as a failed attempt and possibly retry
                download_attempts += 1
                time.sleep(2)  # Wait a bit before retrying
        else:
            print(f"Checksum verification failed for {f}. Retrying...")
            download_attempts += 1
            time.sleep(2)  # Wait a bit before retrying

    if download_attempts >= retry_limit:
        print(f"Failed to download and verify {f} after {retry_limit} attempts.")
        logfile.write(f"Failed to download and verify {f} after {retry_limit} attempts.\n")

    logfile.close()



def download_pubmed(data_dir='.', filter_downloads_by_year=False, years_to_include=['2024']):
    """
    Streamlined method to download PubMed files, optionally filtered by year.
    """
    # Links to the NIH FTP server
    download_config = {
        "ftp": "ftp.ncbi.nlm.nih.gov",
        "baseline": "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
        "update": "ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
    }


    # check data_dir exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Adjust current directory to data_dir
    os.chdir(data_dir)

    # Initialize FTP connection
    ftp = FTP(download_config['ftp'])
    ftp.login()

    # Prepare lists for download and summaries
    to_download = []
    summary = {
        'baseline': {'total': 0, 'xml': 0},
        'update': {'total': 0, 'xml': 0}
    }

    for key, file_dir in [('baseline', '/pubmed/baseline/'), ('update', '/pubmed/updatefiles/')]:
        ftp.cwd(file_dir)
        files = ftp.nlst()

        summary[key]['total'] = len(files)
        summary[key]['xml'] = sum(f.endswith('.xml.gz') for f in files)

        if filter_downloads_by_year:
            year_to_files = separate_files_by_year(files)
            files = filter_files_by_years(year_to_files, years_to_include)

        # Pair up .xml.gz files with their .md5 checksum files
        paired_files = [(f, f + '.md5') for f in files if f.endswith('.xml.gz')]
        to_download.extend([(download_config[key], pair) for pair in paired_files])

    ftp.quit()  # Close the FTP connection

    # Print summary of files to download
    print(f"{summary['baseline']['total']} files in baseline directory. {summary['baseline']['xml']} are .xml.gz files to download.")
    print(f"{summary['update']['total']} files in update directory. {summary['update']['xml']} are .xml.gz files to download.")

    # Download, extract, and verify files
    for base_url, (file, checksum) in to_download[:10]:
        print(f"Downloading and verifying {file} from {base_url}...")
        download_document(file, checksum, base_url)

    print(f"Completed downloading to {data_dir}.")



download_pubmed(data_dir='./data', filter_downloads_by_year=True, years_to_include=['2024'])
