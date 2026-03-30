import urllib.request
import os

os.makedirs("waid_samples", exist_ok=True)

# A few sample aerial wildlife images from public sources
urls = [
    ("https://raw.githubusercontent.com/xiaohuicui/WAID/main/sample/sample1.jpg", "sample1.jpg"),
    ("https://raw.githubusercontent.com/xiaohuicui/WAID/main/sample/sample2.jpg", "sample2.jpg"),
    ("https://raw.githubusercontent.com/xiaohuicui/WAID/main/sample/sample3.jpg", "sample3.jpg"),
]

for url, name in urls:
    path = os.path.join("waid_samples", name)
    try:
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved to {path}")
    except Exception as e:
        print(f"  Failed: {e}")

print("\nDone! Check the waid_samples/ folder.")