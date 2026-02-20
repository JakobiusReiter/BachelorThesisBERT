from transformers import pipeline
import json 
import time
import regex as re 
import emoji, os

BATCH_SIZE = 32
BERT_MODELS_INITIALIZED = False 
MISTRAL_MODELS_INITIALIZED = False

models = ["distilbert_sentiment", "bertweet_sentiment", "roberta_sentiment", "bertweet_emotion", "roberta_language", "roberta_ai"]

modelsToParams = {
    "roberta_language": ["text-classification", "papluca/xlm-roberta-base-language-detection", 512],
    "distilbert_sentiment": ["sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english", 512],
    "bertweet_sentiment": ["text-classification", "finiteautomata/bertweet-base-sentiment-analysis", 128],
    "roberta_sentiment": ["text-classification", "cardiffnlp/twitter-roberta-base-sentiment-latest", 512],
    "emotions": ["text-classification", "finiteautomata/bertweet-base-emotion-analysis", 128],
    "roberta_ai": ["text-classification", "fakespot-ai/roberta-base-ai-text-detection-v1", 512],
}

topicToPrompt = {
    "Minneapolis ICE shooting protests": "protests about a woman (Rene Good) getting killed by an ICE agent in Minneapolis",
    "Trump inauguration":  "Trump's second inauguration as the 47th president of the United States",
    "US attacks Venezuela": "US attacks on Venezuela in which sitting president Nicolas Madura was forcefully extracted to a New York City prison",
    "Zohran Mamdani sworn in": "The inauguration of Zohran Mamdani as the 112th mayor of New York City",
    "Alex Pretti shooting": "The unlawful shooting of nurse Alex Pretti during an anti-ICE protest"
}

topicToPositiveExamples = {
    "Minneapolis ICE shooting protests": "1. Supports the protests/protesters\n2. Is anti-trump or anti-ice or anti-dhs\n3. is pro-immigration",
    "Trump inauguration":  "1. Supports Trump\n2. Is anti-immigration, anti-socialist, anti-communist, or anti-democrat",
    "US attacks Venezuela": "1. Supports the attack, Trump, or ICE\n2. Dislikes Venezuela",
    "Zohran Mamdani sworn in": "1. Supports Zohran Mamdani\n2. Is pro-communist, pro-socialist, or pro-democrat\n3. Is anti-ICE, anti-DHS, or anti-Trump",
    "Alex Pretti shooting": "1. Supports ICE, Trump, or DHS\n2. Blames Pretti for his own death"
}

# replace links with HTTPURL and @ mentions with @USER
def AddSpecialTokens(string): 
    string = re.sub(r"@[^@ ]*", "@USER", string)
    string = re.sub(r"https:[\S]*", "HTTPURL", string)
    return string

# removing empty comments or comments only consisting of stickers
def FilterComments(comments):
    newComments = []
    for comment in comments:
        if comment == "" or comment == "[Sticker]": continue
        newComments.append(AddSpecialTokens(comment))
    return newComments

def InitializeModel(modelName):
    modelParams = modelsToParams[modelName]
    model = pipeline(modelName[0], model=modelParams[1], truncation=True, max_length=modelParams[2])
    return model

# return comments as list (extracted from the sentiment/comment collection)
def GetComments(fileContent, demojified=False):
    comments = []
    for commentContainer in fileContent["content"]:
        if demojified:
            comments.append(commentContainer["comment"]["demojified"])
        else:
            comments.append(commentContainer["comment"]["base"])
    return comments

dataPath = "Data/"
for model in models: 
    model, modelInitialized = False
    for topic in os.listdir(f"{dataPath}Comments"):
        if not os.path.exists(f"{dataPath}Sentiment/{topic}/"): os.mkdir(f"{dataPath}Sentiment/{topic}/") # make dir if not exists
        for fileName in os.listdir(f"{dataPath}Comments/{topic}/"):
            filePath = f"Sentiment/{topic}/{fileName}"
            # dict structure already built, just load it
            if os.path.exists(filePath):
                with open(filePath, "r", encoding="utf-8") as file: 
                    fileContent = json.load(file)
                # skip if already calculated this model
                if model in fileContent["content"][0].keys(): continue
            # build dict structure because this is the first application of any AI on these comments
            else:
                with open(f"Comments/{topic}/{fileName}", "r", encoding="utf-8") as file: 
                    comments = json.load(file)
                fileContent = {}
                fileContent["content"] = []
                for comment in FilterComments(comments["content"]):
                    commentStructure = {
                        "comment": {
                            "base": comment,
                            "demojified": emoji.demojize(comment)
                        },
                        "classification": {}
                    }
                    fileContent["content"].append(commentStructure)
                fileContent["statistics"] = comments["statistics"]
                fileContent["statistics"]["bert_analysis_time"] = 0
                
            platform = fileName.split("-")[0]
            media = fileName.split("-")[1].split(".")[0]
            statistics = fileContent["statistics"]
            
            start = time.time()
            # Init model
            if not modelInitialized:
                model = InitializeModel(model)
            analyzedComments = model(GetComments(fileContent), BATCH_SIZE) 
            analyzedCommentsDemojified = model(GetComments(fileContent, demojified=True), BATCH_SIZE) 
            statistics["bert_analysis_time"] = statistics["bert_analysis_time"] + (time.time() - start)
            for i, classification in enumerate(analyzedComments):
                fileContent["content"]["classification"][i][model] = classification
                fileContent["content"]["classficiation"][i][model + "_demojified"] = analyzedCommentsDemojified[i]

            os.makedirs(f"Data/Classification/{topic}", exist_ok=True)
            with open(f"Data/Classification/{topic}/{fileName}", "w", encoding="utf-8") as file:
                json.dump(fileContent, file, ensure_ascii=False) 