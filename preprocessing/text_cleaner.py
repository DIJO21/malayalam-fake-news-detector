import re
import pandas as pd
from bs4 import BeautifulSoup

class TextCleaner:
    def __init__(self):
        pass
        
    def clean_malayalam_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # 1. Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # 2. Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 3. Remove English characters and numbers (Optional based on project needs, but standard for pure Malayalam analysis)
        text = re.sub(r'[a-zA-Z0-9]', '', text)
        
        # 4. Remove special characters and punctuation
        text = re.sub(r'[^\u0D00-\u0D7F\s]', '', text)
        
        # 5. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess_dataset(self, df, text_column):
        """
        Preprocesses a specified text column in a pandas DataFrame.
        """
        df['cleaned_text'] = df[text_column].apply(self.clean_malayalam_text)
        # Drop rows with empty cleaned text
        df = df[df['cleaned_text'].str.strip().astype(bool)]
        return df

if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "ഇതൊരു പരീക്ഷണ വാചകമാണ്! https://example.com 123 English"
    print(cleaner.clean_malayalam_text(sample))
