import json, os, time
import numpy as np
import matplotlib.pyplot as plt 
from itertools import product

# HYPERPARAMETERS
# which model used for sentiment analysis, multiple different possible - list
#   if multiple models, how to confound their different 'takes', i.e., mean, max confidence...
# whether to ignore which language the comment is in - bool
#   if only english accepted, what is the minimum confidence? - float in [0,1]
# accept / ignore AI comments - bool
#   only accept human comments of which confidence? - float in [0,1]
topicToLeaning = {
    # economic (left/right)
    # political (libertarian/authoritarian)
    "Trump inauguration": [.9, .9],
    "Minneapolis ICE shooting protests": [-.3, -.9],
    "Alex Pretti shooting": [0, 1],
    "Zohran Mamdani sworn in": [-.9, -.7], 
    "US attacks Venezuela": [.3, .7],
    "US attacks Iran": [.3, .7]
}
models = ["distilbert_sentiment", "bertweet_sentiment", "roberta_sentiment", "bertweet_emotion", "roberta_language", "roberta_ai", "mistral_sentiment"]
sentimentModels = ["distilbert_sentiment", "bertweet_sentiment", "roberta_sentiment", "mistral_sentiment"]
choices = ["weighted", "all", "vote", "highest", "sorted"]
modelToModelShort = {
    "distilbert_sentiment": "dist",
    "bertweet_sentiment": "tweet",
    "roberta_sentiment": "roberta",
    "mistral_sentiment": "mistral",
    "bertweet_emotion": "emotion",
    "roberta_language": "language",
    "roberta_ai": "ai"
}
BASE_PATH = os.getcwd()

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

def GetModelScore(dataPoint, model):
    if model == "mistral_sentiment" or model == "mistral_sentiment_demojified":
        return 1
    else: return dataPoint["classification"][model]["score"]

def GetModelLabel(dataPoint, model):
    if model == "mistral_sentiment" or model == "mistral_sentiment_demojified":
        return dataPoint["classification"][model]
    else: return dataPoint["classification"][model]["label"]
         

def CollectCommentLeanings(comments, topic, models, minimumConfidence, demojify, englishOnly, englishConfidence, humanOnly, humanConfidence, choice, ignoreNeutral):
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
            bestPrediction = 1e10
            if choice == "weighted":
                    # maybe not right, definitely not normalized
                    bestPrediction = 0
                    usedModels = 0
                    for model in models:
                        currentModelConfidence = GetModelScore(dataPoint, model)
                        if currentModelConfidence < minimumConfidence: continue
                        currentModelPrediction = GetModelLabel(dataPoint, model)
                        usedModels += 1
                        bestPrediction += currentModelConfidence * currentModelPrediction
                    # print(f"AMOUNT: {len(models)}")                    
                    if len(models) > 3: 
                        print(bestPrediction) 
                        print(usedModels) 
                    if usedModels == 0:    
                        bestPrediction == 1e10
                    else:   
                        bestPrediction /= usedModels
            elif choice == "highest": 
                    for model in models:
                        currentModelConfidence = GetModelScore(dataPoint, model)
                        currentModelPrediction = GetModelLabel(dataPoint, model)
                        if currentModelConfidence < minimumConfidence: continue
                        if currentModelConfidence > highestConfidence:
                            highestConfidence = currentModelConfidence
                            bestPrediction = currentModelPrediction * currentModelConfidence
            elif choice == "sorted":
                    for model in models:
                        currentModelConfidence = GetModelScore(dataPoint, model)
                        if currentModelConfidence > minimumConfidence:
                            bestPrediction = GetModelLabel(dataPoint, model) * currentModelConfidence
                            break
            elif choice == "vote":
                    predictions = {0:0,-1:0,1:0}
                    for model in models: 
                        currentModelConfidence = GetModelScore(dataPoint, model)
                        if currentModelConfidence < minimumConfidence: continue
                        currentModelPrediction = GetModelLabel(dataPoint, model)
                        predictions[currentModelPrediction] += currentModelConfidence
                    bestPrediction = max(predictions, key=predictions.get)
                    if predictions[bestPrediction] == 0: bestPrediction = 1e10 
            elif choice == "all":
                    for model in models:
                        currentModelConfidence = GetModelScore(dataPoint, model)
                        if currentModelConfidence < minimumConfidence: continue
                        bestPrediction = GetModelLabel(dataPoint, model) 
                        commentLeanings.append([bestPrediction*currentModelConfidence*leaning[0],bestPrediction*currentModelConfidence*leaning[1]])
                    continue
            if ignoreNeutral and bestPrediction == 0: 
                continue
            if bestPrediction != 1e10: 
                commentLeanings.append([bestPrediction*leaning[0],bestPrediction*leaning[1]])
    return commentLeanings

def VisualizeData(x, y, fileName, drawLegend, zoom):
    _, ax = plt.subplots()
    if zoom and max(x) < .5 and min(x) > -.5 and max(y) < .5 and min(y) > -.5:
        l = -.5
        r = .5
        gridAmount = 5
    else:
        l = -1
        r = 1
        gridAmount = 9
    ax.set_xlim(l, r)
    ax.set_ylim(l, r)
    ax.set_xticks([l, r])
    ax.set_yticks([l, r])
    ax.set_xticks(np.linspace(l, r, gridAmount), minor=True)
    ax.set_yticks(np.linspace(l, r, gridAmount), minor=True)    
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    labels = ["TikTok", "YouTube", "Instagram", "X", "Facebook"]
    for i, label in enumerate(labels):
        plt.scatter(x[i], y[i], label=label, marker="x")
    if drawLegend: ax.legend()

    filePath = f"{BASE_PATH}/Data/Images/{fileName}.png"
    
    plt.savefig(filePath)
    plt.close()
    return

def CalculateResults(models,
                     minimumConfidence,
                     demojify=True,
                     englishOnly=False, englishConfidence=0.8,
                     humanOnly=False, humanConfidence=0.8,
                     choice="highest", 
                     ignoreNeutral=True,
                     override=True,
                     drawLegend=False,
                     zoom=False):
    start = time.time()
    result = {
        # "used_comments": 0,
        # "parameters": {},
        "results": {
            "TikTok": [],
            "YouTube": [],
            "Instagram": [],
            "X": [],
            "Facebook": []
        },
        "calculation_time": 0,
        "comment_amount": {
            "TikTok": 0,
            "YouTube": 0,
            "Instagram": 0,
            "X": 0,
            "Facebook": 0
        }
    }
    # build file name
    fileName = ""
    for model in models:
        fileName += modelToModelShort[model] + "_"
    if englishOnly: fileName += "language_"
    if humanOnly: fileName += "ai_"
    if demojify: fileName += "demojified_"
    fileName += choice + "_"
    if ignoreNeutral: fileName += "ignoreNeutral_"
    fileName += str(minimumConfidence)
    if zoom: fileName += "_zoomed"
    if not drawLegend: fileName += "_noLegend"

    files = [
        "dist_highest_0.95",
        "tweet_highest_0.95",
        "roberta_highest_0.95",
        "mistral_highest_0.95",
        "dist_demojified_highest_0.95",
        "dist_ai_demojified_highest_0.95_noLegend",
        "dist_language_ai_demojified_highest_0.95_noLegend",
        "tweet_demojified_highest_0.95",
        "tweet_ai_demojified_highest_0.95_noLegend",
        "tweet_language_ai_demojified_highest_0.95_noLegend",
        "roberta_demojified_highest_0.95",
        "roberta_ai_demojified_highest_0.95_noLegend",
        "roberta_language_ai_demojified_highest_0.95_noLegend",
        "mistral_demojified_highest_0.95",
        "mistral_ai_demojified_highest_0.95_noLegend",
        "mistral_language_ai_demojified_highest_0.95_noLegend",
        "dist_tweet_roberta_mistral_language_ai_all_0.95",
        "dist_tweet_roberta_mistral_language_ai_highest_0.95_noLegend",
        "dist_tweet_roberta_mistral_language_ai_weighted_0.95_noLegend",
        "dist_tweet_roberta_mistral_language_ai_vote_0.95_noLegend",
        "dist_tweet_roberta_mistral_language_ai_sorted_0.95_noLegend",
        "dist_tweet_roberta_mistral_language_ai_all_0.5",
        "dist_tweet_roberta_mistral_language_ai_all_0.8_noLegend",
        "dist_tweet_roberta_mistral_language_ai_all_0.9_noLegend",
        "dist_tweet_roberta_mistral_language_ai_all_0.95_noLegend"
    ]
    if fileName not in files:
        return
    
    # currently skipping already created images
    filePath = f"{BASE_PATH}/Data/Images/{fileName}.png"
    if os.path.isfile(filePath):
        if override: os.remove(filePath)
        else: return 
        
    print(f"Creating {fileName}...")

    path = f"{BASE_PATH}/Data/Classification_CONDENSED/"
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
            result["results"][platform].extend(CollectCommentLeanings(fileContent["content"][platform], topic, models=models, minimumConfidence=minimumConfidence, demojify=demojify, englishOnly=englishOnly, englishConfidence=englishConfidence, humanOnly=humanOnly, humanConfidence=humanConfidence, choice=choice, ignoreNeutral=ignoreNeutral))
    result["calculation_time"] = time.time() - start

    for platform in result["results"].keys():
        t = [0,0]
        for comment in result["results"][platform]:
            t = [t[0]+comment[0], t[1]+comment[1]]
        commentAmount = len(result["results"][platform])
        t = [t[0]/commentAmount,t[1]/commentAmount]
        result["results"][platform] = t
        result["comment_amount"][platform] += commentAmount

    with open(f"{BASE_PATH}/Data/Analysis/output.json", "w") as file: 
        json.dump(result, file, ensure_ascii=False)

    xData = []
    yData = []
    for k,v in result["results"].items():
        xData.append(v[0])
        yData.append(v[1]) 
    VisualizeData(xData, yData, fileName, drawLegend, zoom)

from itertools import combinations  
comb = [list(combinations(sentimentModels, r)) for r in range(1, len(sentimentModels) + 1)]  
comb = [list(sublist) for g in comb for sublist in g]
bitmasks = [list(bits) for bits in product([False, True], repeat=6)]

start = time.time()
trueFalse = [True, False]
for modelList in comb:
    for certainty in [.5,.8,.9,.95]:
        for choice in choices: 
            for bitmask in bitmasks:
                override = False
                CalculateResults(modelList, minimumConfidence=certainty, choice=choice, demojify=bitmask[0], englishOnly=bitmask[1], ignoreNeutral=bitmask[2], humanOnly=bitmask[3], override=override, drawLegend=bitmask[4], zoom=bitmask[5])
end = time.time() - start
print(end)