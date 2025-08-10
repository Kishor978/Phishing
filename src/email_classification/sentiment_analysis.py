import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from email_classification.email_utils import setup_logging

# Setup logging
logger = setup_logging()

# Download VADER lexicon if not already downloaded
def download_vader_lexicon():
    try:
        nltk.data.find('vader_lexicon')
        logger.info("VADER lexicon already downloaded")
    except LookupError:
        logger.info("Downloading VADER lexicon...")
        nltk.download('vader_lexicon')
        logger.info("VADER lexicon downloaded successfully")

class PhishingSentimentAnalyzer:
    """
    Analyzes sentiment and emotional triggers commonly used in phishing emails.
    Uses VADER for basic sentiment analysis and custom word lists for specific
    phishing tactics detection.
    """
    def __init__(self):
        download_vader_lexicon()
        self.sia = SentimentIntensityAnalyzer()
        
        # Define word lists for specific phishing emotional triggers
        self.urgency_words = [
            'urgent', 'immediate', 'immediately', 'now', 'quickly', 'today', 
            'soon', 'hurry', 'limited time', 'act now', 'deadline', 'expiring',
            'expires', 'running out', 'last chance', 'final notice', 'alert'
        ]
        
        self.fear_words = [
            'warning', 'alert', 'security', 'breach', 'suspend', 'terminated', 
            'compromise', 'problem', 'risk', 'threat', 'dangerous', 'unauthorized',
            'suspicious', 'illegal', 'fraud', 'violation', 'penalty', 'locked',
            'disabled', 'restricted', 'blocked', 'criminal', 'investigation'
        ]
        
        self.reward_words = [
            'congratulations', 'winner', 'free', 'gift', 'prize', 'offer', 
            'bonus', 'discount', 'exclusive', 'selected', 'special', 'promotion',
            'reward', 'limited offer', 'won', 'claim', 'opportunity', 'deal'
        ]
        
        self.trust_words = [
            'official', 'verified', 'secure', 'guaranteed', 'certified', 
            'trusted', 'legitimate', 'authorized', 'confidential', 'private',
            'important', 'attention', 'notice', 'update', 'confirm', 'validation'
        ]
        
        logger.info("PhishingSentimentAnalyzer initialized with emotional trigger word lists")
    
    def get_emotional_trigger_scores(self, text):
        """Calculate scores for different emotional triggers in text."""
        text_lower = text.lower()
        
        # Calculate trigger scores as proportion of matching words
        urgency_score = sum(1 for word in self.urgency_words if word in text_lower) / len(self.urgency_words)
        fear_score = sum(1 for word in self.fear_words if word in text_lower) / len(self.fear_words)
        reward_score = sum(1 for word in self.reward_words if word in text_lower) / len(self.reward_words)
        trust_score = sum(1 for word in self.trust_words if word in text_lower) / len(self.trust_words)
        
        return {
            'urgency': urgency_score,
            'fear': fear_score,
            'reward': reward_score,
            'trust': trust_score
        }
    
    def analyze(self, subject, body):
        """
        Analyze sentiment and emotional triggers in email subject and body.
        Returns a dictionary of sentiment scores and emotional trigger metrics.
        """
        # Combine subject and body for overall analysis
        full_text = f"{subject} {body}"
        
        # Get VADER sentiment scores
        sentiment_scores = self.sia.polarity_scores(full_text)
        
        # Get emotional trigger scores
        trigger_scores = self.get_emotional_trigger_scores(full_text)
        
        # Subject-specific analysis (typically where urgency is emphasized)
        subject_sentiment = self.sia.polarity_scores(subject)
        subject_triggers = self.get_emotional_trigger_scores(subject)
        
        # Combine all results
        results = {
            # Basic sentiment scores
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu'],
            'positive': sentiment_scores['pos'],
            'compound': sentiment_scores['compound'],
            
            # Overall emotional trigger scores
            'urgency': trigger_scores['urgency'],
            'fear': trigger_scores['fear'],
            'reward': trigger_scores['reward'],
            'trust': trigger_scores['trust'],
            
            # Subject-specific scores
            'subject_negative': subject_sentiment['neg'],
            'subject_compound': subject_sentiment['compound'],
            'subject_urgency': subject_triggers['urgency'],
            
            # Derived metrics for phishing likelihood based on emotion patterns
            'manipulation_score': (trigger_scores['urgency'] + trigger_scores['fear'] + 
                                  trigger_scores['trust'] + sentiment_scores['neg']) / 4
        }
        
        return results
    
    def get_explanation(self, analysis_results):
        """
        Generate human-readable explanations of sentiment analysis results.
        Returns a list of explanation strings based on detected emotional triggers.
        """
        explanations = []
        
        # Check for significant emotional triggers
        if analysis_results['urgency'] > 0.15:
            explanations.append(f"This email uses urgent language (score: {analysis_results['urgency']:.2f}), "
                              f"a common phishing tactic to pressure recipients into acting quickly without thinking.")
        
        if analysis_results['fear'] > 0.15:
            explanations.append(f"This email contains fear-inducing language (score: {analysis_results['fear']:.2f}), "
                              f"a manipulation tactic used to scare recipients into taking immediate action.")
        
        if analysis_results['reward'] > 0.15:
            explanations.append(f"This email promises rewards or special offers (score: {analysis_results['reward']:.2f}), "
                              f"which is often used in phishing to entice recipients to click on malicious links.")
        
        if analysis_results['trust'] > 0.2:
            explanations.append(f"This email uses trust-building language (score: {analysis_results['trust']:.2f}), "
                              f"attempting to appear legitimate by using official-sounding terms.")
        
        if analysis_results['subject_urgency'] > 0.2:
            explanations.append(f"The subject line contains particularly urgent language (score: {analysis_results['subject_urgency']:.2f}), "
                              f"designed to grab attention and create immediate concern.")
        
        if analysis_results['negative'] > 0.2:
            explanations.append(f"This email has a negative emotional tone (score: {analysis_results['negative']:.2f}), "
                              f"which may be used to create anxiety or worry.")
        
        if analysis_results['manipulation_score'] > 0.25:
            explanations.append(f"Overall manipulation score: {analysis_results['manipulation_score']:.2f}. "
                              f"This indicates a combination of emotional triggers typical in phishing attempts.")
        
        # If no significant triggers found
        if not explanations:
            explanations.append("No significant emotional manipulation tactics detected in this email.")
        
        return explanations
