import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class FakeReviewDetector:
    def __init__(self, model_path='./models/roberta_model'):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Add sentiment analyzer for aspects
        self.vader = SentimentIntensityAnalyzer()
        
        # Define product aspects to look for
        self.aspect_keywords = {
            'Battery': ['battery', 'charge', 'charging', 'power', 'battery life'],
            'Camera': ['camera', 'photo', 'picture', 'video', 'lens', 'megapixel'],
            'Screen': ['screen', 'display', 'brightness', 'resolution', 'pixels'],
            'Price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money'],
            'Quality': ['quality', 'build', 'material', 'durable', 'sturdy', 'solid'],
            'Delivery': ['delivery', 'shipping', 'arrived', 'package', 'ship', 'deliver'],
            'Design': ['design', 'look', 'appearance', 'style', 'aesthetic', 'beautiful'],
            'Performance': ['fast', 'slow', 'speed', 'performance', 'lag', 'smooth'],
            'Size': ['size', 'big', 'small', 'compact', 'large', 'tiny', 'fit'],
            'Sound': ['sound', 'audio', 'speaker', 'volume', 'loud', 'noise']
        }
    
    def predict(self, review_text):
        """Original fake detection"""
        inputs = self.tokenizer(review_text, return_tensors='pt', 
                               padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        trust_score = int((1 - probabilities[0][1].item()) * 100)
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': round(confidence * 100, 2),
            'trust_score': trust_score,
            'fake_probability': round(probabilities[0][1].item() * 100, 2),
            'real_probability': round(probabilities[0][0].item() * 100, 2)
        }
    
    def extract_aspects(self, review_text):
        """Extract product aspects and analyze sentiment for each"""
        text_lower = review_text.lower()
        sentences = re.split(r'[.!?]+', review_text)
        
        found_aspects = {}
        
        for aspect_name, keywords in self.aspect_keywords.items():
            aspect_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    aspect_sentences.append(sentence.strip())
            
            if aspect_sentences:
                aspect_text = ' '.join(aspect_sentences)
                sentiment_scores = self.vader.polarity_scores(aspect_text)
                compound = sentiment_scores['compound']
                
                if compound >= 0.05:
                    sentiment = 'Positive'
                    emoji = '😊'
                    color = 'green'
                elif compound <= -0.05:
                    sentiment = 'Negative'
                    emoji = '😞'
                    color = 'red'
                else:
                    sentiment = 'Neutral'
                    emoji = '😐'
                    color = 'gray'
                
                found_aspects[aspect_name] = {
                    'sentiment': sentiment,
                    'emoji': emoji,
                    'color': color,
                    'score': round(compound, 2),
                    'text': aspect_text[:100] + '...' if len(aspect_text) > 100 else aspect_text,
                    'positive': round(sentiment_scores['pos'] * 100, 1),
                    'negative': round(sentiment_scores['neg'] * 100, 1),
                    'neutral': round(sentiment_scores['neu'] * 100, 1)
                }
        
        return found_aspects
    
    def get_overall_sentiment(self, review_text):
        """Get overall sentiment of the review"""
        sentiment_scores = self.vader.polarity_scores(review_text)
        compound = sentiment_scores['compound']
        
        stars = round((compound + 1) * 2.5, 1)
        
        if compound >= 0.05:
            sentiment = 'Positive'
            emoji = '😊'
        elif compound <= -0.05:
            sentiment = 'Negative'
            emoji = '😞'
        else:
            sentiment = 'Neutral'
            emoji = '😐'
        
        return {
            'sentiment': sentiment,
            'emoji': emoji,
            'stars': stars,
            'compound': round(compound, 2),
            'positive': round(sentiment_scores['pos'] * 100, 1),
            'negative': round(sentiment_scores['neg'] * 100, 1),
            'neutral': round(sentiment_scores['neu'] * 100, 1)
        }
    
    def complete_analysis(self, review_text):
        """Complete analysis: fake detection + sentiment + aspects"""
        return {
            'fake_detection': self.predict(review_text),
            'overall_sentiment': self.get_overall_sentiment(review_text),
            'aspects': self.extract_aspects(review_text)
        }
    
    def analyze_reviews(self, reviews_list):
        """Analyze multiple reviews with aspect aggregation"""
        results = [self.predict(review) for review in reviews_list]
        avg_trust = sum(r['trust_score'] for r in results) / len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
        
        all_aspects = {}
        for review in reviews_list:
            aspects = self.extract_aspects(review)
            for aspect_name, aspect_data in aspects.items():
                if aspect_name not in all_aspects:
                    all_aspects[aspect_name] = {
                        'scores': [],
                        'mentions': 0
                    }
                all_aspects[aspect_name]['scores'].append(aspect_data['score'])
                all_aspects[aspect_name]['mentions'] += 1
        
        aspect_summary = {}
        for aspect_name, data in all_aspects.items():
            avg_score = sum(data['scores']) / len(data['scores'])
            
            if avg_score >= 0.05:
                sentiment = 'Positive 😊'
            elif avg_score <= -0.05:
                sentiment = 'Negative 😞'
            else:
                sentiment = 'Neutral 😐'
            
            aspect_summary[aspect_name] = {
                'sentiment': sentiment,
                'avg_score': round(avg_score, 2),
                'mentions': data['mentions'],
                'percentage': round(data['mentions'] / len(reviews_list) * 100, 1)
            }
        
        return {
            'product_trust_score': round(avg_trust, 1),
            'total_reviews': len(results),
            'fake_reviews': fake_count,
            'real_reviews': len(results) - fake_count,
            'fake_percentage': round((fake_count / len(results)) * 100, 1),
            'aspect_summary': aspect_summary
        }