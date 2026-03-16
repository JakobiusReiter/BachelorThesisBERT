import os, json

dataPath = "BachelorThesisBERT/Data"
platforms = ["TikTok", "YouTube", "Instagram", "Facebook", "X"]
platformAmount = {
    "TikTok":0, "YouTube":0, "Instagram":0, "Facebook":0, "X":0
}
topicAmount = {}
all = {
    # topic -> platform -> leaning:amount
}
leaningAmount = {
    "LEFT": 0,
    "CENTER": 0,
    "RIGHT": 0
}
for topic in os.listdir(f"{dataPath}/Classification_CONDENSED/"):
    with open(f"{dataPath}/Classification_CONDENSED/{topic}", "r", encoding="utf-8") as file: 
        data = json.load(file)

    topic = topic.split(".")[0]
    if topic not in all.keys():
        all[topic] = {}

    for platform in data["content"].keys():
        if platform not in all[topic].keys():
            all[topic][platform] = {}
        
        for leaning in data["content"][platform].keys():
            if leaning not in all[topic][platform].keys():
                all[topic][platform][leaning] = 0

            commentList = data["content"][platform][leaning]
            all[topic][platform][leaning] += len(commentList)

with open(f"{dataPath}/counted.json", "w", encoding="utf-8") as file: 
    json.dump(all, file, ensure_ascii=False) 