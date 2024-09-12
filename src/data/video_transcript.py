from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

class YTTranscript:
    def __init__(self, video_id):
        self.video_id = video_id
        self.prefered_language = "English"
        self.default_language_code = "en"
        self.original_language_code = ""
        self.available = True
        self.formatter = TextFormatter()


    def initial_attempt(self):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
            self.original_language_code = list(transcript_list._generated_transcripts.keys())[0]
            return transcript_list
        except:
            self.available = False
            print(f"Transcript not available for the YouTube Video id {self.video_id}")
            return None
    
    def get_transcript(self, transcript_list):
        transcript = transcript_list.find_generated_transcript([self.original_language_code])
        translated_transcript = transcript.translate(self.default_language_code)
        text_formatted = self.formatter.format_transcript(translated_transcript.fetch())

        return text_formatted

    def format_text(self, text_formatted):
        new_text = []
        for _, line in enumerate(text_formatted):
            new_text.append(line.replace("\n", " "))

        new_text = "".join(new_text)
        return new_text
    
    def process(self):
        transcript_list = self.initial_attempt()
        if self.available:
            text_formatted = self.get_transcript(transcript_list)
            formated_text = self.format_text(text_formatted)
            return formated_text
        else:
            return ""


if __name__ == "__main__":
    # video_id = "jx7xuHlfsEQ"
    # video_id  = "ZFhUFazam4A"
    # video_id  = "60SZ70q5IiU"
    # video_id  = "qEyTb7nL0j0"
    # video_id  = "5AzgjUi6xJA"
    video_id = "jx7xuHlfsEQ"

    transcript = YTTranscript(video_id).process()
    print(transcript)
