import shutil
import time
import urllib.request
import re
import tarfile
import os

WORKSPACE_DIR = "./LLM-basic/arxiv/"
DATASET_DIR = "./LLM-basic/dataset/"
QUERY_SIZE = 100


# Function to download and extract a .tar.gz file
def download_and_extract(url):
    # Download the file
    filename = WORKSPACE_DIR + url.split("/")[-1] + ".tar.gz"
    urllib.request.urlretrieve(url, filename)

    # Extract the contents
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=WORKSPACE_DIR, filter="data")

    # Delete the downloaded .tar.gz file
    os.remove(filename)


# Function to find all .tex files
def find_tex_files(directory):
    tex_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
    return tex_files


def get_src_urls(search_url):
    data = urllib.request.urlopen(search_url)
    response = data.read().decode("utf-8")

    # Extract href attributes using regular expression
    hrefs = re.findall(r'<link href="([^"]+)"', response)

    # Throw away the first one, it's not to a paper
    hrefs = hrefs[1:]

    # The ref to download source is found by replacing "/abs/" with "/src/"
    for i in range(len(hrefs)):
        hrefs[i] = hrefs[i].replace("/abs/", "/src/")

    print(f"Extracted {len(hrefs)} Source Hrefs")

    return hrefs


def get_next_dataset_name():
    next_num = len(os.listdir(DATASET_DIR))
    return f"{next_num:08d}.tex"


def add_to_dataset(gzip_url):
    print(f"Getting source for {gzip_url}")
    os.mkdir(WORKSPACE_DIR)

    # Download and extract the file
    download_and_extract(gzip_url)

    # Find all .tex files
    tex_files = find_tex_files(WORKSPACE_DIR)

    # Print the paths of all .tex files found
    print("Found .tex files:")
    for tex_file in tex_files:
        name = get_next_dataset_name()
        print(name, "<--", tex_file)
        os.rename(tex_file, DATASET_DIR + name)

    shutil.rmtree(WORKSPACE_DIR)


if __name__ == "__main__":
    for i in range(100):
        print(f"NOW PROCESSING RESULTS STARTING AT NUMBER {i * QUERY_SIZE}")
        search_url = f"http://export.arxiv.org/api/query?search_query=all:electron&start={i * QUERY_SIZE}&max_results={QUERY_SIZE}"
        src_urls = get_src_urls(search_url)
        time.sleep(4)  # To stay within rate limits

        # URL to the .tar.gz file
        for url in src_urls:
            add_to_dataset(url)
            time.sleep(1)  # To stay within rate limits
