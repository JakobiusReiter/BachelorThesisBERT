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
        time.sleep(random.uniform(1,2))
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
        # time.sleep(random.uniform(1,2))
        # page.locator(":scope > .x5yr21d").click()
        time.sleep(random.uniform(1,2))

        indexes = page.evaluate(
        """() => {
        return Array.from(
            document.querySelectorAll(
            'div.x5yr21d.xw2csxc.x1odjw0f.x1n2onr6'
            )
        ) .map((el, i) => el.scrollHeight > el.clientHeight ? i : null)
        .filter(i => i !== null);
        } """)
        base = page.locator( "div.x5yr21d.xw2csxc.x1odjw0f.x1n2onr6" )
        commentsContainer = [base.nth(i) for i in indexes]
        commentsContainer = commentsContainer[0]
        commentsContainer.evaluate( "el => el.scrollTop = el.scrollHeight" )

        last_height = 0
        iterationCount = 0
        commentTexts = []
        while True:
            iterationCount += 1

            newHeight = Scroll(last_height, commentsContainer)
            if (last_height == newHeight): break
            last_height = newHeight

            if iterationCount >= 10:
                iterationCount = 0
                commentTexts.extend(GetComments(commentsContainer, firstRun))
                if firstRun: firstRun = False
                time.sleep(random.uniform(1,3))
        commentTexts.extend(GetComments(commentsContainer, firstRun))
        page.close()
    return commentTexts
    # StoreComments(link, commentTexts)

# ExecuteCrawl("https://www.instagram.com/p/DS-yA22iony/?hl=de")