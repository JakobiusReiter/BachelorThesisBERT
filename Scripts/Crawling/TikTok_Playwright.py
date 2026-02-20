from playwright.sync_api import sync_playwright
import time
import random
import json

def GetVideoID(link):
    return link.split("/")[-1]

def StoreComments(link, comments):
    with open(GetVideoID(link)+".json", 'w', encoding="utf-8") as myfile:
        json.dump(comments, myfile, ensure_ascii=False)

# scroll comments
def ExecuteCrawl(link):
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()

        page.goto(link, timeout=60000)
        time.sleep(random.uniform(1,2))

        # open comments
        page.locator("span[data-e2e='comment-icon']").nth(0).locator("xpath=..").click()
        time.sleep(random.uniform(1,2))
        # get comment container for scrolling
        commentsContainer = page.locator("div[class*='DivCommentListContainer']")
        commentsScroll = commentsContainer.locator("xpath=..")
        last_height = 0
        while True:
            commentsScroll.evaluate( "(el) => el.scrollTop = el.scrollHeight" )
            time.sleep(random.uniform(1,2))

            # check if new content appeared
            newHeight = commentsScroll.evaluate( "(el) => el.scrollHeight" )
            if newHeight == last_height:
                break
            last_height = newHeight

        # get comments from loaded DOM
        commentLocators = commentsContainer.locator(":scope > *")
        commentTexts = []
        print("Collecting TikTok Comments")
        for i in range(commentLocators.count()):
            commentText = commentLocators.nth(i).locator("span[data-e2e*='comment-level-1']")
            if (commentText.count() <= 0): continue
            txt = commentText.inner_text().strip()
            # print(txt)
            if (txt == ""): continue
            commentTexts.append(txt)
        page.close()
    return commentTexts
    # StoreComments(link, commentTexts)

# ExecuteCrawl("https://www.tiktok.com/@ohcookie.at/video/7586314813752478998")