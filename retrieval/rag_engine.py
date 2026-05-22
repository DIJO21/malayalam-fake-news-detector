import os
import requests
import json
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        try:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except:
            self.openai_client = None
            
        print("Loading Sentence Transformer for Semantic Retrieval...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
    def fetch_live_news(self, query, language='ml'):
        """Fetches live news from NewsData.io based on the query, with fallback search widths."""
        if not self.news_api_key:
            return []
            
        words = query.split()
        if not words:
            return []
            
        # Try different query lengths: 5 words, 3 words, 2 words, 1 word
        for num_words in [5, 3, 2, 1]:
            if len(words) < num_words:
                continue
            search_query = ' '.join(words[:num_words])
            url = f"https://newsdata.io/api/1/news?apikey={self.news_api_key}&q={search_query}"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if results:
                        return results[:5]  # Return top 5
            except Exception as e:
                print(f"Error fetching news (query size {num_words}): {e}")
        return []
        
    def compute_similarity(self, user_text, retrieved_articles):
        """Computes cosine similarity between user text and retrieved articles."""
        if not retrieved_articles:
            return []
            
        user_embedding = self.encoder.encode(user_text, convert_to_tensor=True)
        
        results = []
        for article in retrieved_articles:
            title = article.get('title', '')
            content = article.get('content', '')
            article_text = f"{title}. {content if content else ''}"
            
            if not article_text.strip():
                continue
                
            article_embedding = self.encoder.encode(article_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(user_embedding, article_embedding).item()
            
            results.append({
                'title': title,
                'link': article.get('link', ''),
                'source': article.get('source_id', 'Unknown'),
                'similarity': similarity
            })
            
        # Sort by similarity descending
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return results

    def verify_facts_with_llm(self, user_text, evidence_list):
        """Uses OpenAI to reason about the credibility of the news based on evidence."""
        if not self.openai_client or not evidence_list:
            return "Insufficient live evidence retrieved to perform LLM reasoning."
            
        evidence_text = "\n\n".join([f"Source: {e['source']}\nTitle: {e['title']}" for e in evidence_list[:3]])
        
        prompt = f"""
        You are an elite Fact-Checking AI. Evaluate the following Malayalam/English news snippet against the provided real-time evidence.
        
        User's News Snippet:
        "{user_text}"
        
        Retrieved Live Evidence:
        {evidence_text}
        
        Task:
        Does the live evidence support or contradict the user's news? Is the user's news likely fake or real based on these sources? 
        Provide a concise, 2-3 sentence explanation. Do not use markdown headers, just return plain text.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and objective fact-checking assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Clean fallback when OpenAI API key is rate-limited or out of quota
            evidence_summary = ", ".join([f"'{e['title']}' ({e['source']})" for e in evidence_list[:2]])
            return f"Live search retrieved matching reports from global feeds, including: {evidence_summary}. (Note: Advanced AI analysis is currently running in fallback mode, but web records indicate active coverage on this topic.)"
