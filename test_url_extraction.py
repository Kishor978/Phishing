import re

def test_url_extraction():
    """Test function to verify our URL extraction logic works with different URL formats"""
    
    # Define the updated extraction function
    def extract_urls(text):
        # Multiple patterns for different URL formats
        
        # Standard URLs with http(s) protocol
        standard_pattern = r'https?://(?:www\.)?[a-zA-Z0-9./-]+(?:\?|\#|&)?(?:[a-zA-Z0-9./&=%_+-]+)?'
        
        # URLs without protocol (e.g., example.com/path)
        no_protocol_pattern = r'(?<!\S)(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[a-zA-Z0-9\/_.-]+)?(?:\?[a-zA-Z0-9=%&_.-]+)?'
        
        # URLs enclosed in brackets or parentheses (common in phishing examples)
        bracketed_pattern = r'\[([a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[a-zA-Z0-9\/_.-]+)?(?:\?[a-zA-Z0-9=%&_.-]+)?)\]'
        
        # Find all matches from different patterns
        standard_urls = re.findall(standard_pattern, text)
        no_protocol_urls = re.findall(no_protocol_pattern, text)
        bracketed_urls = re.findall(bracketed_pattern, text)
        
        # Combine all results, removing duplicates
        all_urls = list(set(standard_urls + no_protocol_urls + bracketed_urls))
        
        # For URLs without protocol, add 'http://' for analysis
        for i, url in enumerate(all_urls):
            if not url.startswith(('http://', 'https://')):
                all_urls[i] = 'http://' + url
        
        return all_urls
    
    # Test cases
    test_cases = [
        {
            'name': 'Standard HTTP URL',
            'text': 'Visit our website at http://example.com for more information.',
            'expected': ['http://example.com']
        },
        {
            'name': 'Standard HTTPS URL with path',
            'text': 'Sign in at https://login.example.com/auth to access your account.',
            'expected': ['https://login.example.com/auth']
        },
        {
            'name': 'URL without protocol',
            'text': 'Check out example.com/products for our latest items.',
            'expected': ['http://example.com/products']
        },
        {
            'name': 'URL in square brackets (common in phishing)',
            'text': 'Click here: [secure-link.com/verify?id=12345] to verify your account.',
            'expected': ['http://secure-link.com/verify?id=12345']
        },
        {
            'name': 'Multiple URL types',
            'text': 'Visit http://example.org or go to [phishing-link.com/verify] or check secure.com',
            'expected': ['http://example.org', 'http://phishing-link.com/verify', 'http://secure.com']
        },
        {
            'name': 'Phishing example with bracketed URL',
            'text': 'To verify your identity and restore full access to your account, please click the secure link below: [secure-verification-link.com/verify?id=12345]',
            'expected': ['http://secure-verification-link.com/verify?id=12345']
        },
    ]
    
    # Run the tests
    print("Testing URL Extraction Logic:\n")
    passed = 0
    
    for tc in test_cases:
        print(f"Test: {tc['name']}")
        print(f"Text: {tc['text']}")
        
        # Extract URLs
        extracted = extract_urls(tc['text'])
        extracted.sort()
        expected = tc['expected']
        expected.sort()
        
        print(f"Expected: {expected}")
        print(f"Extracted: {extracted}")
        
        if set(extracted) == set(expected):
            print("✓ PASSED")
            passed += 1
        else:
            print("✗ FAILED")
        print("---")
    
    print(f"\nSummary: {passed}/{len(test_cases)} tests passed")

if __name__ == "__main__":
    test_url_extraction()
