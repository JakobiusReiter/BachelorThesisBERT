import requests
import time

def GetVideoID(link):
    if "watch?v=" in link:
        return link.split("watch?v=")[-1]
    else:
        return link.split("/")[-1]

def CallAPI(link):
    API_KEY = "AIzaSyBjuV2-mcNAAiKCyThSBbcjOSTRGGsQhDQ"
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    params = {
        "part": "snippet",
        "videoId": GetVideoID(link),
        "maxResults": 100,
        "textFormat": "plainText",
        "key": API_KEY
    }

    res = requests.get(url, params=params)
    data = res.json()

    comments = []
    while True:
        res = requests.get(url, params=params)
        data = res.json()

        # collect comments
        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # check if more pages exist
        if "nextPageToken" in data:
            params["pageToken"] = data["nextPageToken"]
            time.sleep(0.1)
        else:
            break

    return comments