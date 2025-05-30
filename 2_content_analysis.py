import json
import re
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import time

def extract_urls(text):
    """Extract URLs from text using regex."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def check_url(url):
    """Check if a URL is valid and accessible."""
    try:
        # Add timeout to avoid hanging on slow responses
        response = requests.head(url, timeout=5, allow_redirects=True)
        return {
            'url': url,
            'status': response.status_code,
            'valid': 200 <= response.status_code < 400
        }
    except requests.RequestException as e:
        return {
            'url': url,
            'status': 'error',
            'valid': False,
            'error': str(e)
        }

def count_words(text):
    """Count words in text, handling markdown and special characters."""
    # Create a copy of text for analysis
    analysis_text = text
    # Remove markdown links but keep the text
    analysis_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', analysis_text)
    # Remove URLs
    analysis_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', analysis_text)
    # Remove special characters and extra whitespace
    analysis_text = re.sub(r'[^\w\s]', ' ', analysis_text)
    analysis_text = re.sub(r'\s+', ' ', analysis_text).strip()
    # Split and count words
    return len(analysis_text.split())

def analyze_content():
    # Load the JSON file
    with open('raw_outputs/content_with_costs.json', 'r') as f:
        data = json.load(f)

    # Process each model's results
    for model, results in data.items():
        print(f"\nAnalyzing {model}...")
        
        for result in results:
            # Create a copy of the result to preserve original
            original_response = result['response']
            
            # Count words from response
            word_count = count_words(original_response)
            result['words_count'] = word_count
            
            # Extract and check URLs from response
            urls = extract_urls(original_response)
            broken_links = []
            
            if urls:
                print(f"\nChecking URLs for {result['id']} ({result['title']}):")
                
                # Check URLs in parallel
                with ThreadPoolExecutor(max_workers=5) as executor:
                    url_results = list(executor.map(check_url, urls))
                
                # Collect broken links
                broken_links = [
                    url_result['url'] 
                    for url_result in url_results 
                    if not url_result['valid']
                ]
                
                # Print URL status
                for url_result in url_results:
                    status = url_result['status']
                    valid = url_result['valid']
                    print(f"  {url_result['url']}: {'✅' if valid else '❌'} (Status: {status})")
            
            # Add broken links to result
            result['broken_links'] = broken_links
            
            print(f"  {result['id']}: {word_count} words, {len(broken_links)} broken links")

    # Save the updated results
    with open('raw_outputs/content_with_analysis.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\n✅ Analysis complete! Results saved to raw_outputs/content_with_analysis.json")

if __name__ == "__main__":
    analyze_content() 