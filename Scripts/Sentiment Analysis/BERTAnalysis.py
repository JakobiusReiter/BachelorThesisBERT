from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json, time, torch
import regex as re 
import emoji, os

BATCH_SIZE = 32
BERT_MODELS_INITIALIZED = False 
MISTRAL_MODELS_INITIALIZED = False

models = ["distilbert_sentiment", "bertweet_sentiment", "roberta_sentiment", "bertweet_emotion", "roberta_language", "roberta_ai", "mistral_sentiment"]
modelsToParams = {
    "roberta_language": ["text-classification", "papluca/xlm-roberta-base-language-detection", 512],
    "distilbert_sentiment": ["sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english", 512],
    "bertweet_sentiment": ["text-classification", "finiteautomata/bertweet-base-sentiment-analysis", 128],
    "roberta_sentiment": ["text-classification", "cardiffnlp/twitter-roberta-base-sentiment-latest", 512],
    "emotions": ["text-classification", "finiteautomata/bertweet-base-emotion-analysis", 128],
    "roberta_ai": ["text-classification", "fakespot-ai/roberta-base-ai-text-detection-v1", 512],
    "bertweet_emotion": ["text-classification", "finiteautomata/bertweet-base-emotion-analysis", 128]
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
    model = pipeline(modelParams[0], model=modelParams[1], truncation=True, max_length=modelParams[2])
    return model

def CreateMistralPrompts(comments, topic):
    prompts = []
    for comment in comments: 
        prompts.append(f"""
            Choose either token P for positive, N for negative or U for uncertain/neutral;
            Only answer with EXACTLY ONE of these tokens: P, N, U;
            DO NOT add punctuation, newlines, or spaces;
            DO NOT explain you answer
            POSITIVE sentiment could include:
            {topicToPositiveExamples[topic]} 
            NEGATIVE sentiment could include the opposite.
            
            What is this comments sentiment about the {topicToPrompt[topic]}? The Comment: '{comment}'""")
    return prompts

def InitializeMistralModel():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    mistralTokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistralTokenizer.pad_token = mistralTokenizer.eos_token
    mistralModel = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        device_map="auto",
        use_safetensors = True
    )
    GetMistralSentiment = (mistralTokenizer, mistralModel)

    return GetMistralSentiment


def MistralAnalyzeComments(model, comments, topic, chunkSize=8):
    mistralPrompts = CreateMistralPrompts(comments, topic)
    mistralTokenizer = model[0]
    mistralModel = model[1]
    mistralSentiment = []
    forced = mistralTokenizer(["P", "N", "U"], add_special_tokens=False).input_ids
    space_id = mistralTokenizer(" ", add_special_tokens=False).input_ids[0]
    nl_id = mistralTokenizer("\n", add_special_tokens=False).input_ids[0]
    genConfig = GenerationConfig(
        max_new_tokens = 1,
        do_sample = False,
        forced_words_ids=[[i] for i in forced],
        bad_words_ids=[[space_id], [nl_id]]
    )
    
    forced = mistralTokenizer(["P", "N", "U"], add_special_tokens=False).input_ids
    for i in range(0, len(mistralPrompts), chunkSize): 
        print(i)
        tempList = mistralPrompts[i:i+chunkSize]
        mistralInputs = mistralTokenizer(tempList, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            mistralOutputs = mistralModel.generate(**mistralInputs, max_new_tokens=5, do_sample=False, generation_config=genConfig, pad_token_id=mistralTokenizer.eos_token_id)
        newSentiment = []
        for j, output in enumerate(mistralOutputs):
            decoded = mistralTokenizer.decode(output, skip_special_tokens=True)
            o = decoded[len(tempList[j]):].strip()
            if o not in ["N", "P"]:
                o = "U"
            newSentiment.append(o)
            
        mistralSentiment.extend(newSentiment)
    return mistralSentiment


# return comments as list (extracted from the sentiment/comment collection)
def GetComments(fileContent, demojified=False):
    comments = []
    for commentContainer in fileContent["content"]:
        if demojified:
            comments.append(commentContainer["comment"]["demojified"])
        else:
            comments.append(commentContainer["comment"]["base"])
    return comments

dataPath = "BachelorThesisBERT/Data/"
for modelName in models: 
    model = False
    modelInitialized = False
    for topic in os.listdir(f"{dataPath}Comments"):
        if not os.path.exists(f"{dataPath}Classification/{topic}/"): os.mkdir(f"{dataPath}Classification/{topic}/") # make dir if not exists
        for fileName in os.listdir(f"{dataPath}Comments/{topic}/"):
            filePath = f"{dataPath}Classification/{topic}/{fileName}"
            # dict structure already built, just load it
            if os.path.exists(filePath):
                with open(filePath, "r", encoding="utf-8") as file: 
                    fileContent = json.load(file)
                # skip if already calculated this model 
                if modelName in fileContent["content"][0]["classification"].keys(): 
                    print(f"{modelName} Analysis already done on {topic}-{fileName}")
                    continue
            # build dict structure because this is the first application of any AI on these comments
            else:
                with open(f"{dataPath}Comments/{topic}/{fileName}", "r", encoding="utf-8") as file: 
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
            
            
            # Init model
            if not modelInitialized:
                modelInitialized = True
                if modelName == "mistral_sentiment":
                    model = InitializeMistralModel()
                else:
                    model = InitializeModel(modelName)
            
            # analyze comments
            start = time.time()
            if modelName == "mistral_sentiment":
                analyzedComments = MistralAnalyzeComments(model, CreateMistralPrompts(GetComments(fileContent), topic), topic)
                analyzedCommentsDemojified = MistralAnalyzeComments(model, CreateMistralPrompts(GetComments(fileContent, demojified=True), topic), topic)
                statistics["mistral_analysis_time"] = (time.time() - start)
            else:
                analyzedComments = model(GetComments(fileContent), batch_size=BATCH_SIZE) 
                analyzedCommentsDemojified = model(GetComments(fileContent, demojified=True), batch_size=BATCH_SIZE) 
                statistics["bert_analysis_time"] = statistics["bert_analysis_time"] + (time.time() - start)
            
            # store comments in dict
            print(fileName)
            for i, classification in enumerate(analyzedComments):
                fileContent["content"][i]["classification"][modelName] = classification
                fileContent["content"][i]["classification"][modelName + "_demojified"] = analyzedCommentsDemojified[i]
            # convert dict to json file
            os.makedirs(f"{dataPath}/Classification/{topic}", exist_ok=True)
            with open(f"{dataPath}/Classification/{topic}/{fileName}", "w", encoding="utf-8") as file:
                json.dump(fileContent, file, ensure_ascii=False) 