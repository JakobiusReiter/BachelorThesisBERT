from playwright.sync_api import sync_playwright
import time
import random
import json

def BuildCommentText(comment):
    commentTextContainer = comment.locator("div.xdj266r.x14z9mp.xat24cr.x1lziwak.x1vvkbs").nth(0)
    if commentTextContainer.count() == 0:  
        return ""
    commentSubtextContainers = commentTextContainer.locator(":scope > *")
    tempText = ""
    
    commentSubtextContainersCount = commentSubtextContainers.count()
    for x in range(commentSubtextContainersCount):
        commentSubtextContainer = commentSubtextContainers.nth(x).locator(":scope > *") 
        if commentSubtextContainer.count() == 0:
            tempText += commentSubtextContainers.nth(x).inner_text()
            continue
        commentSubtextContainerCount = commentSubtextContainer.count()

        for y in range(commentSubtextContainerCount):
            commentSubtext = commentSubtextContainer.nth(y) 
            if commentSubtext.evaluate("el => el.tagName") == "SPAN":
                tempText += commentSubtext.locator(":scope > *").nth(0).get_attribute("alt")
            else: tempText += commentSubtext.inner_text()
    return tempText

def CollectComments(postType, commentsContainer, commentTexts): 
    comments = commentsContainer.locator(":scope > *") 
    commentCount = comments.count()  
    print(commentCount)
    for i in range(commentCount): 
        comment = comments.nth(i) 
        if postType == "photo" or postType == "reel" or postType == "video":
            seeMoreButton = comment.get_by_role("button", name="See more")
            if seeMoreButton.count() > 0:
                seeMoreButton.nth(0).click()
        newText = BuildCommentText(comment)
        if newText.strip() == "":
            continue
        commentTexts.append(newText)
        print(newText)
    return commentTexts

def Scroll(postType, scrollContainer, page): 
    newHeight = 0
    if postType == "post":
        for i in range(10):
            scrollContainer.evaluate( "el => el.scrollTop = el.scrollHeight" )
            time.sleep(random.uniform(.25,.75))
            newHeight = scrollContainer.evaluate( "el => el.scrollHeight" )
    else:
        viewMoreCommentsButton = page.get_by_role("button", name="View more comments") 
        for i in range(10):
            # if button.count() < 1: break
            
            if (viewMoreCommentsButton.count() == 0): return -1
            try:
                viewMoreCommentsButton.click() 
            except:
                return -1
            scrollContainer.evaluate( "el => el.scrollTop = el.scrollHeight" )
            time.sleep(random.uniform(.05,.2))
            newHeight = scrollContainer.evaluate( "el => el.scrollHeight" ) 
    time.sleep(random.uniform(1,3)) 
    return newHeight

def GetScrollContainer(postType, page):
    if postType == "post": 
        scrollContainer = page.locator(
            "div.xb57i2i.x1q594ok.x5lxg6s.x78zum5.xdt5ytf.x6ikm8r.x1ja2u2z.x1pq812k.x1rohswg.xfk6m8.x1yqm8si.xjx87ck.xx8ngbg.xwo3gff.x1n2onr6.x1oyok0e.x1odjw0f.x1iyjqo2.xy5w88m"
        ).nth(0)
    elif postType == "video":
        scrollContainer = page.locator(
            "div.x78zum5.xdt5ytf.x6ikm8r.x1odjw0f.x1iyjqo2.xv54qhq.xf7dkkf"
        ).nth(0)
    else: # photo
        scrollContainer = page.locator(
            "div.xb57i2i.x1q594ok.x5lxg6s.x78zum5.xdt5ytf.x6ikm8r.x1ja2u2z."
            "x1pq812k.x1rohswg.xfk6m8.x1yqm8si.xjx87ck.x1l7klhg.x1iyjqo2."
            "xs83m0k.x2lwn1j.xx8ngbg.xwo3gff.x1oyok0e.x1odjw0f.x1n2onr6.xq1qtft"
        ).nth(0)
    return scrollContainer 

def DeleteComments(commentsContainer, page):
    comments = commentsContainer.locator(":scope > *")
    commentCount = comments.count() 
    for i in range(commentCount): 
        # if i < 2: continue
        if i > commentCount - 3: break
        comments.nth(0).evaluate("el => el.remove()") 

def GetCommentsContainer(postType, page):
    time.sleep(3)
    if postType == "post": 
        commentsContainer = page.locator("div.html-div.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x1gslohp").nth(0)
    elif postType == "video":
        commentsContainer = GetScrollContainer(postType, page)
    else: # photo 
        c = GetScrollContainer(postType, page)
        # commentsContainer = c.locator(":scope > *").nth(0).locator(":scope > *").nth(2).locator(":scope > *").nth(0)
        commentsContainer = c.locator("div.html-div.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl")
        i = 0
        while i < commentsContainer.count():
            print(f"{i} BEN")
            if commentsContainer.nth(i).locator(":scope > *").count() > 5:
                return commentsContainer.nth(i)
            i+=1
    return commentsContainer

def ExecuteCrawl(link):
    with sync_playwright() as p: 
        done = False
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()
        page.goto(link, timeout=60000)
        time.sleep(random.uniform(1,2))

        if "facebook.com/photo" in link:
            postType = "photo" 
        elif "/watch?v=" in link:
            postType = "video"
        elif "/reel/" in link:
            postType = "reel"
        else:
            postType = "post"

        if postType == "reel":
            page.locator("div[aria-label='Comment']").first.click()
            time.sleep(1) 
        
        scrollContainer = GetScrollContainer(postType, page)
        commentsContainer = GetCommentsContainer(postType, page)
        commentTexts = []  
        strike = False
        while len(commentTexts) < 2500 and not done: 
            if postType == "post":
                firstComment = commentsContainer.locator(":scope > *").first
                if firstComment.get_attribute("data-virtualized") != "false":
                    if strike:
                        break 
                    strike = True
                    time.sleep(5)
                    continue
                commentText = BuildCommentText(firstComment)
                commentTexts.append(commentText)
                print(commentText)
                firstComment.evaluate("el => el.remove()") 
                time.sleep(random.uniform(.2,.4))
                strike = False
                continue
            newHeight = Scroll(postType, scrollContainer, page) 
            time.sleep(.5)
            commentTexts = CollectComments(postType, commentsContainer, commentTexts) 
            DeleteComments(commentsContainer, page)
            if (newHeight == 0 or newHeight == -1): break 
        page.close()
        
    return commentTexts