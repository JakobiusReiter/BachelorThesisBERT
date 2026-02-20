import TikTok_Playwright as tiktok
import Instagram_Playwright as instagram
import Twitter_Playwright as twitter
import YouTube_API as youtube_api
import YouTube_Playwright as youtube

import time
import os
import pandas as pd
import json

# chrome --remote-debugging-port=9222 --user-data-dir="C:\chrome-playwright"




platforms = [twitter, instagram, tiktok, youtube]
platformStrings = ["X", "Instagram", "TikTok", "YouTube"]
dataPath = "Data/"
matrix = pd.read_csv(dataPath+"Posts.csv", dtype=str, delimiter=";").to_numpy()

for i, row in enumerate(matrix):
    title = row[0]
    media = row[1]
    path = dataPath + "Comments/" + title
    if not os.path.exists(path):
        os.makedirs(path)

    for col in range(2,6):
        allComments = dict.fromkeys(("content", "statistics"), "")
        allComments["statistics"] = dict.fromkeys(("amount", "crawl_time"), "")

        currentPlatform = platformStrings[col-2]
        filePath = f"{path}/{currentPlatform}-{media}.json"
        if os.path.exists(filePath): continue
        start = time.time()
        link = row[col]
        if link == "NF": continue
        print(currentPlatform)
        if currentPlatform == "YouTube":
            comments = youtube_api.CallAPI(link)
        else: 
            comments = platforms[col-2].ExecuteCrawl(link)
        allComments["content"] = comments
        allComments["statistics"]["crawl_time"] = time.time() - start
        allComments["statistics"]["amount"] = len(comments)
        with open(filePath, 'w', encoding="utf-8") as myfile:
            json.dump(allComments, myfile, ensure_ascii=False)