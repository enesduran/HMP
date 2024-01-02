import os
import warnings
import requests
import os.path as op

# warnings.filterwarnings("ignore", message="Unverified HTTPS request")

def download(hmp_url="https://download.is.tue.mpg.de/download.php?domain=hmp&resume=1&sfile=model.zip", 
             pymaf_url="https://download.is.tue.mpg.de/download.php?domain=hmp&resume=1&sfile=data.zip", 
             body_model_url="https://download.is.tue.mpg.de/download.php?domain=hmp&resume=1&sfile=body_models.zip"):
    
    # Define the username and password
    username = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    
    out_folder = "downloads"

    post_data = {"username": username, "password": password}

    urls = [hmp_url, pymaf_url, body_model_url]

    # Strip newline characters from the URLs
    urls = [url.strip() for url in urls]

    # Loop through the URLs and download the files

    for url in urls:
        print("Downloading", url)
        # Make a POST request with the username and password
        response = requests.post(url,
            data=post_data,
            stream=True,
            verify=False,
            allow_redirects=True)
        
        if response.status_code == 401:
            raise Exception("Authentication failed for URLs. Username/password correct?")
        elif response.status_code == 403:
            raise Exception("You are not authorized to download these files")

        # Get the filename from the URL
        filename = url.split("file=")[1]

        # Write the contents of the response to a file
        out_p = op.join(out_folder, filename)
        
        os.makedirs(op.dirname(out_p), exist_ok=True)
        
        with open(out_p, "wb") as f:
            f.write(response.content)

    print("Done")

if __name__ == "__main__":
    download()
