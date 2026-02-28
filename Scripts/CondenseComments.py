import os, json

"""
This document serves to condense the entirety of the collected, already classified comments into topic files.
Internally, the structure looks like
{
    "content": {
        platform: {
            leaning: [
                comment1: {
                    "content":
                    "classification":
                }
            ] 
        }
    }
    "statistics": {

    }
}
"""

mediaToLeaning = {
    "CNN": "CENTER",
    "ORF": "CENTER",
    "FOX": "RIGHT",
    "MSNBC": "LEFT"
}

def NormalizeComments(content):
    normalizedContent = content

    for comment in content:
        for model in comment["classification"].keys():
            if model == "mistral_sentiment" or model == "mistral_sentiment_demojified":
                label = comment["classification"][model]
                if label == "P":
                    comment["classification"][model] = 1
                elif label == "N":
                    comment["classification"][model] = -1
                else:
                    comment["classification"][model] = 0
                continue
            label = comment["classification"][model]["label"]
            if label in ["positive", "POSITIVE", "POS"]:
                label = comment["classification"][model]["label"] = 1
            elif label in ["negative", "NEGATIVE", "NEG"]:
                label = comment["classification"][model]["label"] = -1
            elif label in ["neutral", "NEUTRAL", "NEU"]:
                label = comment["classification"][model]["label"] = 0

    return normalizedContent

for topic in os.listdir("BachelorThesisBERT/Data/Classification/"):
    allComments = {
        "content": {
            "X": {},
            "Instagram": {},
            "TikTok": {},
            "YouTube": {}
        },
        "statistics": {
            "amount": 0,
            "crawl_time": 0, 
            "bert_analysis_time": 0,
            "mistral_analysis_time": 0
        }
    }
    
    for fileName in os.listdir(f"BachelorThesisBERT/Data/Classification/{topic}/"):
        with open(f"BachelorThesisBERT/Data/Classification/{topic}/{fileName}", "r", encoding="utf-8") as file: 
            fileContent = json.load(file)
        platform = fileName.split(sep="-")[0]
        media = fileName.split(sep="-")[1].split(sep=".")[0]

        print(fileName)
        statistics = fileContent["statistics"]
        
        for key in statistics.keys(): 
            if isinstance(statistics[key], str): statistics[key] = 0
            if key == "time" and key not in allComments["statistics"].keys():
                allComments["statistics"]["crawl_time"] += statistics[key]
                continue
            allComments["statistics"][key] += statistics[key]

        leaning = mediaToLeaning[media]
        fileContent["content"] = NormalizeComments(fileContent["content"])
        if leaning in allComments["content"][platform]:
            allComments["content"][platform][leaning].extend(fileContent["content"])
        else:
            allComments["content"][platform][leaning] = fileContent["content"]
    
    with open(f"BachelorThesisBERT/Data/Classification_CONDENSED/{topic}.json", "w", encoding="utf-8") as file:
        json.dump(allComments, file, ensure_ascii=False)