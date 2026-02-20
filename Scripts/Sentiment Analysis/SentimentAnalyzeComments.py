from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
import json 
import time
import regex as re 
import emoji, os, torch

BATCH_SIZE = 16
BERT_MODELS_INITIALIZED = False 
MISTRAL_MODELS_INITIALIZED = False

topicToPrompt = {
    "Minneapolis ICE shooting protests": "protests about a woman (Rene Good) getting killed by an ICE agent in Minneapolis",
    "Trump inauguration":  "Trump's second inauguration as the 47th president of the United States",
    "US attacks Venezuela": "US attacks on Venezuela in which sitting president Nicolas Madura was forcefully extracted to a New York City prison",
    "Zohran Mamdani sworn in": "The inauguration of Zohran Mamdani as the 112th mayor of New York City"
}

ben = {
    "Minneapolis ICE shooting protests": "1. Supports the protests/protesters\n2. Is anti-trump or anti-ice or anti-dhs\n3. is pro-immigration",
    "Trump inauguration":  "1. Supports Trump\n2. Is anti-immigration, anti-socialist, anti-communist, or anti-democrat",
    "US attacks Venezuela": "1. Supports the attack, Trump, or ICE\n2. Dislikes Venezuela",
    "Zohran Mamdani sworn in": "1. Supports Zohran Mamdani\n2. Is pro-communist, pro-socialist, or pro-democrat\n3. Is anti-ICE, anti-DHS, or anti-Trump"
}

def AddSpecialTokens(string): 
    string = re.sub(r"@[^@ ]*", "@USER", string)
    string = re.sub(r"https:[\S]*", "HTTPURL", string)
    return string

def FilterComments(comments):
    newComments = []
    for comment in comments:
        if comment == "" or comment == "[Sticker]": continue
        newComments.append(AddSpecialTokens(comment))
    return newComments

def InitializeBERTModels(): 
    DetectLanguage = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", truncation=True, max_length=512)
    GetDistilbertSentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512)
    GetBertTweetSentiment = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis", truncation=True, max_length=128)
    GetRobertaSentiment = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True, max_length=512)
    GetEmotions = pipeline("text-classification", model="finiteautomata/bertweet-base-emotion-analysis", truncation=True, max_length=128)
    DetectAI = pipeline("text-classification", model="fakespot-ai/roberta-base-ai-text-detection-v1", truncation=True, max_length=512)

    global BERT_MODELS_INITIALIZED
    BERT_MODELS_INITIALIZED = True

    return DetectLanguage, GetDistilbertSentiment, GetBertTweetSentiment, GetRobertaSentiment, GetEmotions, DetectAI

def InitializeMistralModels():
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

def CreateMistralPrompts(comments, topic):
    prompts = []
    for comment in comments: 
        prompts.append(f"""
            Choose either token P for positive, N for negative or U for uncertain/neutral;
            Only answer with EXACTLY ONE of these tokens: P, N, U;
            DO NOT add punctuation, newlines, or spaces;
            DO NOT explain you answer
            POSITIVE sentiment could include:
            {ben[topic]} 
            NEGATIVE sentiment could include the opposite.
            
            What is this comments sentiment about the {topicToPrompt[topic]}? The Comment: '{comment}'""")
    return prompts

def BERTAnalyzeComments(comments, topic): 
    if not BERT_MODELS_INITIALIZED:
        global DetectLanguage, GetDistilbertSentiment, GetBertTweetSentiment, GetRobertaSentiment, GetEmotions, DetectAI
        DetectLanguage, GetDistilbertSentiment, GetBertTweetSentiment, GetRobertaSentiment, GetEmotions, DetectAI = InitializeBERTModels()

    result = []
    commentsDemojified = [emoji.demojize(comment) for comment in comments] 
    language = DetectLanguage(comments, batch_size=BATCH_SIZE)
    bertTweetSentiment = GetBertTweetSentiment(comments, batch_size=BATCH_SIZE)
    bertTweetSentimentDemojified = GetBertTweetSentiment(commentsDemojified, batch_size=BATCH_SIZE)
    distilbertSentiment = GetDistilbertSentiment(comments, batch_size=BATCH_SIZE)
    distilbertSentimentDemojified = GetDistilbertSentiment(commentsDemojified, batch_size=BATCH_SIZE)
    robertaSentiment = GetRobertaSentiment(comments, batch_size=BATCH_SIZE)
    robertaSentimentDemojified = GetRobertaSentiment(commentsDemojified, batch_size=BATCH_SIZE)
    emotions = GetEmotions(comments, batch_size=BATCH_SIZE)
    emotionsDemojified = GetEmotions(commentsDemojified, batch_size=BATCH_SIZE)
    ai = DetectAI(comments, batch_size=BATCH_SIZE)

    

    for i, comment in enumerate(comments):  
        currentCommentDict = {
            "comment": {
                "base": comment,
                "demojified": commentsDemojified[i]
            },
            "sentiment": {
                "distilbert": distilbertSentiment[i],
                "distilbert_demojified": distilbertSentimentDemojified[i],
                "bertweet": bertTweetSentiment[i],
                "bertweet_demojified": bertTweetSentimentDemojified[i],
                "roberta": robertaSentiment[i],
                "roberta_demojified": robertaSentimentDemojified[i]
            },
            "emotions": {
                "bertweet": emotions[i],
                "bertweet_demojified": emotionsDemojified[i]
            }, 
            "lang": language[i],
            "AI_written": ai[i]
        }
        result.append(currentCommentDict)
    return result

def MistralAnalyzeComments(comments, topic, demojified):
    chunkSize = 8
    global MISTRAL_MODELS_INITIALIZED
    if not MISTRAL_MODELS_INITIALIZED:
        MISTRAL_MODELS_INITIALIZED = True
        global GetMistralSentiment
        GetMistralSentiment = InitializeMistralModels() 
        
    mistralPrompts = CreateMistralPrompts(comments, topic)
    mistralTokenizer = GetMistralSentiment[0]
    mistralModel = GetMistralSentiment[1]
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
            print(o)
            newSentiment.append(o)
            
        mistralSentiment.extend(newSentiment)
    return mistralSentiment


for topic in os.listdir("Comments/"):
    for fileName in os.listdir(f"Comments/{topic}/"):
        if os.path.exists(f"Sentiment/{topic}/{fileName}"): continue
        # if topic != "Minneapolis ICE shooting protests": continue
        platform = fileName.split("-")[0]
        media = fileName.split("-")[1].split(".")[0]  
        # print(f"Analyzing {topic} on {platform} posted by {media}")

        with open(f"Comments/{topic}/{fileName}", "r", encoding="utf-8") as file: 
            fileContent = json.load(file)
        comments = FilterComments(fileContent["content"])
        statistics = fileContent["statistics"]
        
        start = time.time()
        bertAnalyzedComments = BERTAnalyzeComments(comments, topic) 
        statistics["bert_analysis_time"] = time.time() - start
        
        result = {"content": bertAnalyzedComments, "statistics": statistics}
        os.makedirs(f"Sentiment/{topic}", exist_ok=True)
        with open(f"Sentiment/{topic}/{fileName}", "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False)

for topic in os.listdir("Sentiment/"):
    for fileName in os.listdir(f"Sentiment/{topic}/"):
        with open(f"Sentiment/{topic}/{fileName}", "r", encoding="utf-8") as file: 
            fileContent = json.load(file)
        
        start = time.time()
        if "mistral" not in fileContent["content"][0]["sentiment"].keys():
            comments = [d["comment"]["base"] for d in fileContent["content"]]
            mistralAnalyzedComments = MistralAnalyzeComments(comments, topic, False)
            for i in range(0, len(mistralAnalyzedComments)):
                fileContent["content"][i]["sentiment"]["mistral"] = mistralAnalyzedComments[i]
        if "mistral_demojified" not in fileContent["content"][0]["sentiment"].keys():
            commentsDemojified = [d["comment"]["base"] for d in fileContent["content"]]
            mistralAnalyzedCommentsDemojified = MistralAnalyzeComments(commentsDemojified, topic, True)
            for i in range(0, len(mistralAnalyzedCommentsDemojified)): 
                fileContent["content"][i]["sentiment"]["mistral_demojified"] = mistralAnalyzedCommentsDemojified[i]
            fileContent["statistics"]["mistral_analysis_time"] = time.time() - start
        
        with open(f"Sentiment/{topic}/{fileName}", "w", encoding="utf-8") as file:
            json.dump(fileContent, file, ensure_ascii=False)