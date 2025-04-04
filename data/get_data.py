import requests
import re
import argparse


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





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--out_file", type=str, default="urls.txt")

    args = parser.parse_args()

    get_livecell_urls(args)