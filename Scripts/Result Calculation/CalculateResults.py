import json, os, time
import pandas as pd
import numpy as np

# HYPERPARAMETERS
# which model used for sentiment analysis, multiple different possible - list
#   if multiple models, how to confound their different 'takes', i.e., mean, max confidence...
# whether to ignore which language the comment is in - bool
#   if only english accepted, what is the minimum confidence? - float in [0,1]
# accept / ignore AI comments - bool
#   only accept human comments of which confidence? - float in [0,1]
topicToLeaning = {
    "Trump inauguration": [.9, .8],
    "US attacks Venezuela": [.7, .9], 
    "Zohran Mamdani sworn in": [-.9, -.8], 
    "Minneapolis ICE shooting protests": [-.9, -.7],
    "Alex Pretti shooting": [.4,1]
}
models = ["distilbert_sentiment", "bertweet_sentiment", "roberta_sentiment", "bertweet_emotion", "roberta_language", "roberta_ai"]

def CalculateLeaning(negative, neutral, positive, topic):
    leaning = np.array([0.0,0.0])
    commentAmount = negative + neutral + positive
    topicLeaning = topicToLeaning[topic]

    
    leaning += positive * np.array(topicLeaning)
    leaning -= negative * np.array(topicLeaning)
    if commentAmount != 0:
        leaning /= (commentAmount * len(topicToLeaning.keys()))
    else: return [0,0]
    return leaning 

def CollectCommentLeanings(comments, topic, models, minimumConfidence, demojify, englishOnly, englishConfidence, humanOnly, humanConfidence, choice, fillNeutral,ignoreNeutral):
    """
    Docstring for CollectValidComments
    
    :param comments: sentiment analyzed comments (fileContent["content"][platform])
    :param models: models as string descriptors to be used
    :param minimumConfidence: minimum score predictions have to have to be used when scoring
    :param demojify: use demojified predictions
    :param englishOnly: enables filtering by lang, see englishConfidence
    :param englishConfidence: minimum score of lang with label "en" to be used
    :param humanOnly: enables filtering by ai_written, see humanConfidence
    :param humanConfidence: minimum score of ai_writte with label "Human" to be used
    :param choice: "average": returns average predictions of all passed models
                 "highest": returns label of model with highest score
                 "sorted": returns label of first model in list, if above certain score, otherwise next model etc.
                 "vote": returns label which has highest summed score of all models
    """
    leaning = topicToLeaning[topic]
    commentLeanings = []

    if demojify: models = [model+"_demojified" for model in models]
    for spectrum in comments:
        for dataPoint in comments[spectrum]:
            if englishOnly:
                if dataPoint["classification"]["roberta_language_demojified"]["label"] != "en" or dataPoint["classification"]["roberta_language"]["score"] < englishConfidence: continue
            if humanOnly:
                if dataPoint["classification"]["roberta_ai_demojified"]["label"] != "Human" or dataPoint["classification"]["roberta_ai"]["score"] < humanConfidence: continue
            highestConfidence = 0
            bestPrediction = "NEU"
            
            match choice:
                case "weighted":
                    break
                case "highest": 
                    bestPrediction = "undefined"
                    for model in models:
                        currentModelConfidence = dataPoint["classification"][model]["score"]
                        currentModelPrediction = dataPoint["classification"][model]["label"]
                        if currentModelConfidence > highestConfidence:
                            highestConfidence = currentModelConfidence
                            bestPrediction = currentModelPrediction 
                case "sorted":
                    bestPrediction = "undefined"
                    for model in models:
                        currentModelConfidence = dataPoint["classification"][model]["score"]
                        if currentModelConfidence > minimumConfidence:
                            bestPrediction = dataPoint["classification"][model]["label"]
                            break
                case "vote":
                    bestPrediction = "undefined"
                    predictions = {0:0,-1:0,1:0}
                    for model in models:
                        currentModelPrediction = dataPoint["classification"][model]["label"]
                        currentModelConfidence = dataPoint["classification"][model]["score"]
                        predictions[currentModelPrediction] += currentModelConfidence
                    bestPrediction = max(predictions, key=predictions.get)
                case "all":
                    for model in models:
                        bestPrediction = dataPoint["classification"][model]["label"]
                        commentLeanings.append([bestPrediction*leaning[0],bestPrediction*leaning[1]])
            if bestPrediction == "undefined":
                if fillNeutral:
                    bestPrediction = 0
                else:
                    continue
            commentLeanings.append([bestPrediction*leaning[0],bestPrediction*leaning[1]])
    # print(commentLeanings)
    return commentLeanings

def CalculateResults(models,
                     minimumConfidence,
                     demojify=True,
                     englishOnly=False, englishConfidence=0.8,
                     humanOnly=False, humanConfidence=0.8,
                     choice="highest",
                     fillNeutral=False,
                     ignoreNeutral=True):
    start = time.time()
    result = {
        # "used_comments": 0,
        # "parameters": {},
        "results": {
            "TikTok": [],
            "YouTube": [],
            "Instagram": [],
            "X": [],
        },
        "calculation_time": 0,
        "comment_amount": {
            "TikTok": 0,
            "YouTube": 0,
            "Instagram": 0,
            "X": 0,
        }
    }

    path = "Data/Classification_CONDENSED/"
    for topic in os.listdir(path):
        topic=topic.split(".")[0]
        with open(f"{path}/{topic}.json", "r", encoding="utf-8") as file: 
            fileContent = json.load(file)
        
        for platform in fileContent["content"]: 
            # cull left / right to keep even
            if "LEFT" in fileContent["content"][platform].keys(): 
                leftAmount = len(fileContent["content"][platform]["LEFT"])
            else: leftAmount = 0
            if "RIGHT" in fileContent["content"][platform].keys(): 
                rightAmount = len(fileContent["content"][platform]["RIGHT"])
            else: rightAmount = 0
            # print(f"left: {leftAmount}, right: {rightAmount}")
            if rightAmount > leftAmount:
                fileContent["content"][platform]["RIGHT"] = fileContent["content"][platform]["RIGHT"][:leftAmount]
            elif leftAmount > rightAmount: 
                fileContent["content"][platform]["LEFT"] = fileContent["content"][platform]["LEFT"][:rightAmount]

            # print(f"{topic} --- {platform}")
            result["results"][platform].extend(CollectCommentLeanings(fileContent["content"][platform], topic, models=models, minimumConfidence=minimumConfidence, demojify=demojify, englishOnly=englishOnly, englishConfidence=englishConfidence, humanOnly=humanOnly, humanConfidence=humanConfidence, choice=choice, fillNeutral=fillNeutral, ignoreNeutral=ignoreNeutral))
    result["calculation_time"] = time.time() - start

    for platform in result["results"].keys():
        t = [0,0]
        for comment in result["results"][platform]:
            t = [t[0]+comment[0], t[1]+comment[1]]
        commentAmount = len(result["results"][platform])
        t = [t[0]/commentAmount,t[1]/commentAmount]
        result["results"][platform] = t
        result["comment_amount"][platform] += commentAmount

    with open(f"Data/Analysis/output.json", "w") as file: 
        json.dump(result, file, ensure_ascii=False)
CalculateResults(["roberta_sentiment", "distilbert_sentiment", "bertweet_sentiment"], .9, demojify=True, choice="vote")
# CalculateResults(["mistral"], .9, demojify=False, choice="highest") mistral not implemented yet