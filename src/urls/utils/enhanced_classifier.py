import re
import math
import string
import json
import os
from urllib.parse import urlparse
from collections import Counter
import torch

from urls.models import CharCNN
from urls.utils.urls_preprocessing import encode_url_char, MAX_URL_LEN

class EnhancedURLClassifier:
    """
    Enhanced URL classifier that combines CharCNN model predictions with rule-based 
    heuristics and a domain whitelist for improved accuracy.
    """
    def __init__(self, model_path=None, vocab_path=None):
        """
        Initialize the enhanced URL classifier.
        
        Args:
            model_path: Path to the trained CharCNN model (.pt file)
            vocab_path: Path to the character vocabulary JSON file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and vocabulary
        self.model = None
        self.char_vocab = None
        
        # Load model and vocabulary if paths are provided
        if model_path and vocab_path and os.path.exists(model_path) and os.path.exists(vocab_path):
            self._load_model_and_vocab(model_path, vocab_path)
        
        # Initialize whitelist of known legitimate domains
        self.whitelist = self._load_whitelist()
    
    def _load_model_and_vocab(self, model_path, vocab_path):
        """Load the CharCNN model and vocabulary."""
        try:
            # Load vocabulary
            with open(vocab_path, 'r') as f:
                self.char_vocab = json.load(f)
            
            # Initialize model
            self.model = CharCNN(vocab_size=len(self.char_vocab) + 1, embed_dim=64).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            print(f"CharCNN model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def _load_whitelist(self):
        """Load the whitelist of known legitimate domains."""
        return [
            "google.com", "github.com", "microsoft.com", "amazon.com", "netflix.com", 
            "apple.com", "facebook.com", "youtube.com", "twitter.com", "instagram.com",
            "linkedin.com", "reddit.com", "wikipedia.org", "stackoverflow.com", 
            "yahoo.com", "twitch.tv", "spotify.com", "dropbox.com", "gmail.com",
            "outlook.com", "office.com", "live.com", "hotmail.com", "paypal.com",
            "aol.com", "wordpress.com", "wix.com", "adobe.com", "ebay.com",
            "cloudflare.com", "shopify.com", "godaddy.com", "baidu.com", "apache.org",
            "oracle.com", "salesforce.com", "ibm.com", "hp.com", "intel.com",
            "cisco.com", "mozilla.org", "ubuntu.com", "python.org"
        ]
    
    def is_known_legitimate(self, url):
        """Check if URL belongs to a known legitimate domain."""
        try:
            parsed = urlparse(url if '://' in url else 'http://' + url)
            domain = parsed.netloc or url.split('/', 1)[0]
            
            # Extract the base domain (e.g., github.com from sub.github.com)
            domain_parts = domain.split('.')
            if len(domain_parts) > 2:
                base_domain = '.'.join(domain_parts[-2:])
            else:
                base_domain = domain
                
            return base_domain.lower() in self.whitelist
        except:
            return False
    
    def extract_url_features(self, url):
        """
        Extract detailed features from a URL for better classification
        
        Features extracted:
        - Length features
        - Domain-specific features
        - Character distribution
        - Special pattern detection
        - TLD analysis
        
        Returns:
            dict: Dictionary of extracted features
        """
        # Basic length features
        features = {
            'url_length': len(url),
            'domain_length': 0,
            'path_length': 0,
            'num_dots': url.count('.'),
            'num_digits': sum(c.isdigit() for c in url),
            'num_special': sum(c in string.punctuation for c in url),
        }
        
        # Parse URL components
        try:
            parsed = urlparse(url if '://' in url else 'http://' + url)
            domain = parsed.netloc or url.split('/', 1)[0]  # Fallback if parsing fails
            path = parsed.path
            
            features['domain_length'] = len(domain)
            features['path_length'] = len(path)
            features['has_path'] = len(path) > 0
            features['path_depth'] = path.count('/')
            
            # TLD analysis
            tld_match = re.search(r'\.([a-z]{2,})$', domain)
            features['tld'] = tld_match.group(1) if tld_match else ""
            features['is_common_tld'] = features['tld'] in ['com', 'org', 'net', 'edu', 'gov', 'io']
            
            # Special pattern detection
            features['has_suspicious_keywords'] = any(kw in domain.lower() for kw in [
                'secure', 'verify', 'account', 'login', 'update', 'alert', 'confirm', 'wallet', 'password'
            ])
            
            # Statistical features
            features['digit_ratio'] = features['num_digits'] / len(url) if len(url) > 0 else 0
            features['special_ratio'] = features['num_special'] / len(url) if len(url) > 0 else 0
            
            # Domain entropy (randomness measure)
            char_counts = Counter(domain)
            domain_len = len(domain)
            entropy = -sum((count / domain_len) * math.log2(count / domain_len) for count in char_counts.values())
            features['domain_entropy'] = entropy
            
            # Suspicious patterns
            features['has_multiple_subdomains'] = domain.count('.') > 1
            features['has_misleading_protocol'] = 'https' in domain and not domain.startswith('https')
            features['domain_dash_count'] = domain.count('-')
            features['domain_underscore_count'] = domain.count('_')
            
        except Exception as e:
            print(f"Error parsing URL '{url}': {str(e)}")
        
        return features
    
    def classify_url(self, url, model_prediction=None, model_confidence=None):
        """
        Enhanced URL classifier that combines ML model prediction with rule-based heuristics
        
        Args:
            url: URL to classify
            model_prediction: Optional pre-computed CharCNN model prediction ("Phishing" or "Legitimate")
            model_confidence: Optional pre-computed model confidence score
            
        Returns:
            tuple: (final_prediction, confidence, reason)
        """
        # Step 1: Check whitelist first for quick legitimate classification
        if self.is_known_legitimate(url):
            return "Legitimate", 0.95, "Domain in trusted whitelist"
        
        # Step 2: Get model prediction if not provided
        if model_prediction is None or model_confidence is None:
            if self.model is not None:
                # Preprocess URL for CharCNN input
                char_encoded = encode_url_char(url, max_len=MAX_URL_LEN, char2idx=self.char_vocab)
                char_input = torch.tensor(char_encoded, dtype=torch.long).unsqueeze(0).to(self.device) # Add batch dimension

                # Get model prediction
                with torch.no_grad():
                    output = self.model(char_input)
                    model_confidence = output.item()  # Get probability

                model_prediction = "Phishing" if model_confidence > 0.5 else "Legitimate"
            else:
                # If no model is available, rely only on heuristics
                model_prediction = "Unknown"
                model_confidence = 0.5
        
        # Step 3: Extract features for rule-based analysis
        features = self.extract_url_features(url)
        
        # Step 4: Apply heuristic rules with weights
        heuristic_score = 0.0
        
        # Length-based rules (longer URLs more suspicious)
        if features['url_length'] > 100:
            heuristic_score += 0.3
        elif features['url_length'] > 50:
            heuristic_score += 0.15
        
        # Keyword-based rules
        if features['has_suspicious_keywords']:
            heuristic_score += 0.25
        
        # Special character rules
        if features['domain_dash_count'] > 2:
            heuristic_score += 0.2
        
        # TLD rules
        if not features['is_common_tld']:
            heuristic_score += 0.15
        
        # Multiple subdomains
        if features['has_multiple_subdomains']:
            heuristic_score += 0.1
        
        # Misleading patterns
        if features['has_misleading_protocol']:
            heuristic_score += 0.4
        
        # Entropy check (random-looking domains are suspicious)
        if features['domain_entropy'] > 4.0:
            heuristic_score += 0.2
        
        # Step 5: Combine model and heuristics (weighted average)
        model_weight = 0.6  # Give 60% weight to the model
        heuristic_weight = 0.4  # Give 40% weight to the heuristics
        
        # Convert model prediction to score (1 for phishing, 0 for legitimate)
        model_score = float(model_confidence) if model_prediction == "Phishing" else 1.0 - float(model_confidence)
        
        # Calculate final score
        final_score = (model_weight * model_score) + (heuristic_weight * heuristic_score)
        
        # Make final decision
        if final_score >= 0.5:
            return "Phishing", final_score, "Combined model and rule-based analysis"
        else:
            return "Legitimate", 1.0 - final_score, "Combined model and rule-based analysis"
