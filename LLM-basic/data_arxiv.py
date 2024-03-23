import shutil
import time
import urllib.request
import urllib.error
import re
import tarfile
import os

WORKSPACE_DIR = "./LLM-basic/arxiv/"
# a bit of jank to get around GuildAI trying to copy all our training data for every run
DATASET_DIR = "/home/green/Coding/ML-Experiments/LLM-basic/dataset/"
QUERY_SIZE = 100
# TOPICS:
# electron up to 3335
# economics up to 2948
TOPIC = "economics"
STARTING_POINT = 2948 + 1


# Function to download and extract a .tar.gz file
def download_and_extract(url):
    # Download the file
    filename = WORKSPACE_DIR + url.split("/")[-1] + ".tar.gz"
    try:
        urllib.request.urlretrieve(url, filename)
    except urllib.error.HTTPError as e:
        print("HTTP error, skipping file: ", e)
        return

    # Extract the contents
    try:
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=WORKSPACE_DIR, filter="data")
    except tarfile.ReadError as e:
        print("Read error, skipping file: ", e)

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


def load_dataset():
    dataset = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".tex"):
                file_path = os.path.join(root, file)
                # Try multiple encodings
                encodings = ["utf-8", "latin-1"]  # You can extend this list as needed
                content = None
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                        break  # Break out of the loop if file is successfully read
                    except UnicodeDecodeError:
                        pass  # Try the next encoding
                if content is not None:
                    dataset.append(content)
                else:
                    print(f"Unable to read file: {file_path}")
    return dataset


if __name__ == "__main__":
    for i in range(100):
        search_url = f"http://export.arxiv.org/api/query?search_query=all:{TOPIC}&start={i * QUERY_SIZE + STARTING_POINT}&max_results={QUERY_SIZE}"
        src_urls = get_src_urls(search_url)

        if len(src_urls) == 0:
            print("Got no results, terminating")
            break

        print("Sleeping for 4 seconds...")
        time.sleep(4)  # To stay within rate limits

        # URL to the .tar.gz file
        for j, url in enumerate(src_urls):
            print(f"NOW PROCESSING URL {i * QUERY_SIZE + j + STARTING_POINT}")
            add_to_dataset(url)
            print("Sleeping for 4 seconds...")
            time.sleep(4)  # To stay within rate limits
