from playwright.sync_api import sync_playwright
import time
import random
import re
import json

def GetPostID(link):
    return link.split("/")[-1]

def StoreComments(link, comments):
    with open(GetPostID(link)+".json", 'w', encoding="utf-8") as myfile:
        json.dump(comments, myfile, ensure_ascii=False)

def GetComments(page, highestOffset, allComments):
    comments_container = page.locator("div[aria-label='Timeline: Conversation']").locator("div").first
    time.sleep(random.uniform(1,2))
    comments = comments_container.locator(":scope > *").all()

    for c in comments:
        match = re.search("\d+", c.get_attribute("style").split(";")[0])
        value = int(match.group(0))
        childCount = c.locator(":scope > *").locator(":scope > *").count()
        if (value == 0 or childCount <= 0 or value <= highestOffset): continue
        
        txt = c.locator("div[data-testid='tweetText']")
        if (txt.count() <= 0): continue
        children = txt.locator(":scope > *")
        childCount = children.count()
        completeText = ""
        for i in range(childCount):
            child = children.nth(i)
            tagName = child.evaluate("el => el.tagName.toLowerCase()")
            if tagName == "span":
                completeText += child.text_content()
            elif tagName == "img":
                completeText += child.get_attribute("alt")
        allComments.append(completeText)
    highestOffset = value
    return highestOffset

def ExecuteCrawl(link):
    allComments = [] 
    highestOffset = -1

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")

        context = browser.contexts[0]
        page = context.new_page()
        page.goto(link)
        time.sleep(random.uniform(1,2))

        previousHighestOffset = 0
        while (previousHighestOffset != highestOffset):
            previousHighestOffset = highestOffset
            highestOffset = GetComments(page, highestOffset, allComments)
            page.mouse.wheel(0,3000)
            time.sleep(random.uniform(1,2))
        
        page.close()
    return allComments
        # StoreComments(link, allComments)

# ExecuteCrawl("https://x.com/FoxNews/status/2006820181213925533")