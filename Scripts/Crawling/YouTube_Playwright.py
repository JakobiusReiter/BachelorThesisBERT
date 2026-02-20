from playwright.sync_api import sync_playwright
import time
import random
import json

def GetVideoID(link):
    return link.split("/")[-1]

def StoreComments(link, comments):
    with open(GetVideoID(link)+".json", 'w', encoding="utf-8") as myfile:
        json.dump(comments, myfile)

# scroll comments
def ExecuteCrawl(link):
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()

        page.goto(link, timeout=60000)
        time.sleep(random.uniform(1,2))
        
        page.locator("button[aria-label*='comments']").click()
        time.sleep(random.uniform(1,2))

        comments_container = page.locator("ytd-item-section-renderer#sections > #contents").first
        last_height = 0
        while True:
            page.evaluate(
                "(el) => el.scrollTop = el.scrollHeight",
                comments_container.element_handle()
            )
            time.sleep(random.uniform(.5, 1))

            # check if new content appeared
            new_height = page.evaluate(
                "(el) => el.scrollHeight",
                comments_container.element_handle()
            )

            if new_height == last_height:
                break
            last_height = new_height


        # get comments from loaded DOM
        comments = page.locator( "yt-attributed-string#content-text" ).all()
        comment_texts = [
            c.inner_text().strip()
            for c in comments
        ]

        # store comments
        page.close()
        return comment_texts
        # StoreComments(link, comment_texts)