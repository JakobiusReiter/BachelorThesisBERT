from playwright.sync_api import sync_playwright
import time
import random
import json

def GetPostType(link):
    if "/reels/" in link:
        return "reel"
    else:
        return "normal"

def GetComments(commentsContainer, firstRun=False):
    # get comments from loaded DOM
    commentTexts = []
    print("get comments")
    for i in range(commentsContainer.count()):
        print(i)
        if not firstRun and i in range(10): continue
        commentLocator = commentsContainer.nth(i).locator("div.html-div.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x9f619.xjbqb8w.x78zum5.x15mokao.x1ga7v0g.x16uus16.xbiv7yw.x1uhb9sk.x1plvlek.xryxfnj.x1c4vz4f.x2lah0s.xdt5ytf.xqjyukv.x1cy8zhl.x1oa3qoh.x1nhvcw1")
        if (commentLocator.count() < 1): continue
        commentText = commentLocator.first.inner_text()
        print(commentText)
        commentTexts.append(commentText)
    print("delete dom comments")
    for i in range(commentsContainer.count()-10):
        commentsContainer.nth(0).evaluate("(el) => el.remove()")
    return commentTexts

def Scroll(oldHeight, commentsContainer):
    newHeight = oldHeight
    for i in range(100):
        commentsContainer.evaluate( "(el) => el.scrollTop = el.scrollHeight" )
        time.sleep(random.uniform(.05,.1))
        newHeight = commentsContainer.evaluate( "(el) => el.scrollHeight" )
        if newHeight != oldHeight:
            return newHeight
    return newHeight

def GetCommentsContainer(postType, page, scrollContainer): 

    if postType == "reel":
        commentsContainer = scrollContainer.locator(":scope > *").nth(1).locator(":scope > *")
    else:
        commentsContainer = scrollContainer.locator(":scope > *").nth(0).locator(":scope > *").nth(2).locator(":scope > *")
    return commentsContainer

def GetScrollContainer(postType, page):
    
    if postType == "reel":
        scrollContainer = page.locator(
            "div.html-div.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.x18d9i69.x9f619.xjbqb8w.x78zum5.x15mokao.x1ga7v0g.x16uus16.xbiv7yw.xsfy40s.x1mfogq2.x1uhb9sk.xw2csxc.x1odjw0f.x1iyjqo2.x2lwn1j.xeuugli.xdt5ytf.xqjyukv.x1qjc9v5.x1oa3qoh.x1nhvcw1"
        ).nth(0)
    else: 
        scrollContainer = page.locator(
            "div.x5yr21d.xw2csxc.x1odjw0f.x1n2onr6"
        ).nth(0)
    return scrollContainer

# scroll comments
def ExecuteCrawl(link):
    with sync_playwright() as p:
        firstRun = True
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()

        page.goto(link, timeout=60000) 
        time.sleep(random.uniform(3,5))

        postType = GetPostType(link) 
        print(postType)

        if postType == "reel":
            c = page.locator("svg[aria-label='Kommentar']").first 
            c.click()
            time.sleep(1) 
        scrollContainer = GetScrollContainer(link, page)
        commentsContainer = GetCommentsContainer(link, page, scrollContainer)
        scrollContainer.evaluate( "el => el.scrollTop = el.scrollHeight" )

        last_height = 0
        iterationCount = 0
        commentTexts = []
        while True:
            iterationCount += 1

            newHeight = Scroll(last_height, scrollContainer)
            if (last_height == newHeight): 
                print("newheight was equal to oldheight")
                break
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