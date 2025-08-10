# Phishing Detection System

This project implements an advanced phishing detection system with two main components:
1. Email phishing detection using a dual-encoder transformer architecture
2. URL phishing detection using a hybrid approach with CharCNN model and rule-based enhancement

## Features

### Email Phishing Detection
- Dual-encoder transformer architecture to separately process email subject and body
- Attention fusion mechanism for combining subject and body representations
- Sentiment analysis for detecting emotional manipulation tactics common in phishing
- Automatic URL extraction and analysis

### URL Phishing Detection
- Character-level CNN for learning URL patterns
- Domain whitelist system to prevent false positives for known legitimate domains
- Advanced feature engineering with heuristic rules
- Hybrid classification combining ML model predictions with rule-based analysis

## Recent Enhancements

### Enhanced URL Classification
The URL detection system uses a streamlined approach that combines:

1. Character-level CNN model for effective pattern recognition
2. Domain knowledge whitelist for high-profile legitimate domains
3. Rule-based heuristics analyzing URL features:
   - Length-based features
   - Character distribution and entropy
   - TLD analysis
   - Suspicious keyword detection
   - Special character patterns

This hybrid approach significantly improves accuracy for well-known legitimate domains while maintaining high detection rates for phishing URLs.