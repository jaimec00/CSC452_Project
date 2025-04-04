import requests
import re
import argparse
import zipfile
import io
from tqdm import tqdm

def get_livecell_urls(args):

    '''
    gets the urls contiaining the live cell data.
    '''

    # Define the URL and parameters
    url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/"
    params = {"list-type": "2"}

    # Define the headers as in the curl command
    headers = {
        "Host": "livecell-dataset.s3.eu-central-1.amazonaws.com",
        "Date": "20161025T124500Z",
        "Content-Type": "text/plain"
    }

    # Perform the GET request
    response = requests.get(url, headers=headers, params=params)

    # Save the response content (as text) to a variable
    output = response.text

    # Use regex to extract all content between <Key> and the next '<'
    pattern = r"(?<=<Key>)[^<]+"
    matches = re.findall(pattern, output)

    # Prepend the base URL to each extracted key
    base_url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/"
    urls = [base_url + match for match in matches]

    # If you need a single string (like the content of urls.txt) with each URL on a new line:
    urls_text = "\n".join(urls)

    with open(args.out_file, "w") as f:
        f.write(urls_text)


def get_livecell_images(args):
    # URL of the zip file to download
    zip_url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images_per_celltype.zip"

    # Download the zip file with streaming enabled
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()

    # Determine total size in bytes for the progress bar
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # 1 KB chunks

    # Create a BytesIO object to store the downloaded content
    zip_buffer = io.BytesIO()

    # Set up the progress bar for the download
    with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading") as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive chunks
                zip_buffer.write(chunk)
                progress_bar.update(len(chunk))

    # Rewind the BytesIO object to the beginning
    zip_buffer.seek(0)

    # Open the zip file
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        # Get the list of files in the archive for the extraction progress bar
        file_list = zip_ref.infolist()
        # Set up the progress bar for the extraction process
        with tqdm(total=len(file_list), desc="Extracting") as extract_bar:
            for file_info in file_list:
                zip_ref.extract(file_info, path=args.out_dir)
                extract_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--out_file", type=str, default="urls.txt")
    parser.add_argument("--out_dir", type=str, default="images")
    parser.add_argument("--get_images", type=int, default=1)

    args = parser.parse_args()

    if args.get_images:
        get_livecell_images(args)
    else:
        get_livecell_urls(args)