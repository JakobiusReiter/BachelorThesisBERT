from playwright.sync_api import sync_playwright
import time
import random
import json

def GetComments(commentsContainer, firstRun=False):
    # get comments from loaded DOM
    commentLocators = commentsContainer.locator(":scope > *").first.locator(":scope > *").nth(2).locator(":scope > *")
    commentTexts = []
    print("get comments")
    for i in range(commentLocators.count()):
        if not firstRun and i in range(10): continue
        commentText = commentLocators.nth(i).locator('span[style="--x---base-line-clamp-line-height: 18px; --x-lineHeight: 18px;"]')
        if (commentText.count() != 3): continue
        commentTexts.append(commentText.nth(2).inner_text().strip())
    print("delete dom comments")
    for i in range(commentLocators.count()-10):
        commentLocators.nth(0).evaluate("(el) => el.remove()")
    return commentTexts

def Scroll(oldHeight, commentsContainer):
    newHeight = oldHeight
    for i in range(50):
        commentsContainer.evaluate( "(el) => el.scrollTop = el.scrollHeight" )
        time.sleep(random.uniform(.05,.1))
        newHeight = commentsContainer.evaluate( "(el) => el.scrollHeight" )
        if newHeight != oldHeight:
            return newHeight
    return newHeight

# scroll comments
def ExecuteCrawl(link):
    with sync_playwright() as p:
        firstRun = True
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()
        page.goto(link, timeout=60000)

        time.sleep(random.uniform(1,2))

        commentsContainer = page.locator(
            "div.xb57i2i.x1q594ok.x5lxg6s.x78zum5.xdt5ytf.x6ikm8r.x1ja2u2z."
            "x1pq812k.x1rohswg.xfk6m8.x1yqm8si.xjx87ck.x1l7klhg.x1iyjqo2."
            "xs83m0k.x2lwn1j.xx8ngbg.xwo3gff.x1oyok0e.x1odjw0f.x1n2onr6.xq1qtft"
        ).nth(0)
        viewMoreCommentsSpan = page.get_by_role("button", name="View more comments")
        commentsContainer.evaluate( "el => el.scrollTop = el.scrollHeight" )
        viewMoreCommentsSpan.click()
        
        
        # oldHeight = 0
        # iterationCount = 0
        commentTexts = [] 
        # while True: 
        #     iterationCount += 1

        #     newHeight = Scroll(oldHeight, commentsContainer)
        #     if (oldHeight == newHeight): break
        #     oldHeight = newHeight
        #     viewMoreCommentsSpan.click()

        commentsContainer = page.locator(
            "div.xb57i2i.x1q594ok.x5lxg6s.x78zum5.xdt5ytf.x6ikm8r.x1ja2u2z."
            "x1pq812k.x1rohswg.xfk6m8.x1yqm8si.xjx87ck.x1l7klhg.x1iyjqo2."
            "xs83m0k.x2lwn1j.xx8ngbg.xwo3gff.x1oyok0e.x1odjw0f.x1n2onr6.xq1qtft"
        ).nth(0)
        commentsParent = commentsContainer.locator(":scope > *").nth(0).locator(":scope > *").nth(2).locator(":scope > *").nth(0)
        oldHeight = 0
        commentTexts = [] 
        iterationCount = 1
        while len(commentTexts) < 5000:
            if iterationCount == 0: break
            iterationCount = 0

            while iterationCount < 3: 
                newHeight = Scroll(oldHeight, commentsContainer)
                if (oldHeight == newHeight): break
                oldHeight = newHeight
                viewMoreCommentsSpan.click()
                iterationCount += 1
            
            comments = commentsParent.locator(":scope > *")

            commentCount = comments.count()
            # comments.nth(0).evaluate("(el) => el.remove()")
            # comments.nth(0).evaluate("(el) => el.remove()")
            for i in range(commentCount):
                if i < 2: continue
                if i > commentCount - 3: break
                comment = comments.nth(i)
                seeMoreButton = comment.get_by_role("button", name="See more")
                if seeMoreButton.count() > 0:
                    seeMoreButton.nth(0).click()
                commentText = comment.locator("div.x1lliihq.xjkvuk6.x1iorvi4").nth(0)
                if commentText.count() == 0: continue
                commentText = comment.locator("div.x1lliihq.xjkvuk6.x1iorvi4").nth(0).inner_text()
                print(f"----- {i} -----")
                print(commentText)
                commentTexts.append(commentText)
                
            commentCount = comments.count()
            for i in range(commentCount):
                if i < 2: continue
                if i > commentCount - 3: break
                comments.nth(i).evaluate("(el) => {el.innerHTML = '';}")
            # time.sleep(60)

    return commentTexts

# ExecuteCrawl("https://www.facebook.com/photo.php?fbid=1297321642257695")