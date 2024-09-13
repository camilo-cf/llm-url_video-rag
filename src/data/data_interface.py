from pytube import extract

class URLHandler:
    def __init__(self, urls):
        self.urls_list = list(set(urls.replace(" ", "").split("\n")))
        self.youtube = [self.extract_video_id(each_url)
                        for each_url in self.urls_list
                        if self.is_youtube(each_url)
                        and each_url!=""]
        self.website = [each_url
                        for each_url in self.urls_list
                        if not self.is_youtube(each_url)
                        and each_url!=""]

    def is_youtube(self, url):
        if "wiki" in url:
            return False
        try:
            extract.video_id(url)
            return True
        except:
            return False
    
    def extract_video_id(self, url):
        return extract.video_id(url)

if __name__ == "__main__":
    url_links = "https://docs.smith.langchain.com/old/user_guide \nhttps://www.youtube.com/watch?v=jx7xuHlfsEQ"
    urls = URLHandler(urls=url_links)
    yt = urls.youtube
    web = urls.website

    print("YT:", yt)
    print("WEBSITE:", web)
