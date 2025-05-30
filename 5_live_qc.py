import os
import time
import json
import logging
from datetime import datetime

import openai
from google import genai
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QCResult(BaseModel):
    verdict: str
    tip: str

def check_api_keys():
    """Check if required API keys are set."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is not set")

def generate_article(tip=None, max_retries=3):
    """Generate article content using OpenAI."""
    prompt = {
        "title": "Introduction to Python Object Oriented Programming",
        "keywords": [
            "python", "object oriented programming", "OOP",
            "classes", "objects", "inheritance", "polymorphism", "encapsulation"
        ]
    }
    
    user_instruction = (
        f"Title: \"{prompt['title']}\"\n"
        "- Write a 1,500 - 2,500 word article.\n"
        "- Cite at least two reputable sources with URLs.\n"
        f"- Cover these keywords: {', '.join(prompt['keywords'])}.\n"
        "- Avoid unsafe, biased, or sensitive content.\n"
        "- Use a warm, conversational tone with a light joke or analogy."
        f"\n- {tip}" if tip else ""
    )
    
    system_instruction = (
        "You are an expert tech writer with a friendly, witty personality. "
        "Produce fact-checked, safe, and highly accurate articles."
    )

    for attempt in range(max_retries):
        try:
            response = openai.responses.create(
                model="o4-mini",
                input=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction}
                ],
            )
            
            if response and response.output_text:
                logger.info(f"Successfully generated article on attempt {attempt + 1}")
                return response.output_text
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    logger.error("Failed to generate article after all retries")
    return None

def check_quality(content, max_retries=3):
    """Check content quality using Gemini."""
    if not content:
        logger.error("Empty content provided for quality check")
        return None

    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    for attempt in range(max_retries):
        try:
            user_msg = f"""
            You are a content quality checker. Analyze this draft and provide a JSON response with:
            1. A verdict: "APPROVED" if the content is clear, factual, and safe (all scores ≥4), otherwise "REJECTED"
            2. A brief tip for improvement if rejected
            
            Format your response exactly like this JSON:
            {{
                "verdict": "APPROVED" or "REJECTED",
                "tip": "your improvement suggestion"
            }}
            
            Draft:
            {content}
            """

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=user_msg,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": QCResult,
                },
            )
            
            if response and response.parsed:
                logger.info(f"Successfully performed quality check on attempt {attempt + 1}")
                return response.parsed

        except Exception as e:
            logger.error(f"Quality check attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    logger.error("Failed to perform quality check after all retries")
    return None

def save_results(content, qc_result, status):
    """Save the results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "content": content,
        "qc_result": qc_result.dict(),
        "status": status
    }
    
    os.makedirs("qc_results", exist_ok=True)
    filename = f"qc_results/article_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {filename}")

def main():
    try:
        # Check API keys
        check_api_keys()
        
        # Generate initial content
        logger.info("Generating initial article...")
        content = generate_article()
        if not content:
            logger.error("Failed to generate initial content")
            return
        
        # Perform quality check
        logger.info("Performing quality check...")
        qc_result = check_quality(content)
        if not qc_result:
            logger.error("Failed to perform quality check")
            return
        
        # Process results
        if qc_result.verdict == "APPROVED":
            logger.info("Article approved on first attempt")
            save_results(content, qc_result, "approved")
        else:
            logger.info(f"Article rejected: {qc_result.tip}")
            # Generate revised content
            logger.info("Generating revised article...")
            revised_content = generate_article(tip=qc_result.tip)
            if revised_content:
                # Check revised content
                revised_qc = check_quality(revised_content)
                if revised_qc:
                    status = "approved_revised" if revised_qc.verdict == "APPROVED" else "rejected_revised"
                    save_results(revised_content, revised_qc, status)
                    logger.info(f"Revised article {status}")
            else:
                logger.error("Failed to generate revised content")
                save_results(content, qc_result, "rejected_no_revision")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()