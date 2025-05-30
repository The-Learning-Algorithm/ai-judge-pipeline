import os
import time
import json

import openai
from google import genai
from google.genai import types

# 1) CONFIGURE YOUR CLIENTS
openai.api_key     = os.getenv("OPENAI_API_KEY")
gemini_client      = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2) PRICING TUPLES: (input $/1M, output $/1M, words_per_token)
pricing = {
    # "o4-mini":                     (1.10,  4.40, 0.80),
    # "chatgpt-4o-latest":                    (5.00, 15.00, 0.75),
    # "o3-2025-04-16":               (10.00, 40.00, 0.75),
    # "gemini-2.5-pro-preview-05-06": (1.25, 10.00, 0.70),
    "gemini-2.5-flash-preview-05-20":            (0.15, 3.50, 0.70),
}

# 3) PROMPTS: titles + keywords
prompts = [
    {
        "id": "P1",
        "title": "5 Trends in AI-Assisted Learning",
        "keywords": [
            "personalized learning", "adaptive quizzes", "AI tutoring",
            "learning analytics", "student engagement"
        ]
    },
    {
        "id": "P2",
        "title": "Remote Work: The New Normal",
        "keywords": [
            "hybrid teams", "virtual collaboration", "digital nomads",
            "work-life balance", "productivity tools"
        ]
    },
    {
        "id": "P3",
        "title": "Blockchain Adoption in Small Business",
        "keywords": [
            "supply chain", "smart contracts", "transaction fees",
            "decentralization", "security"
        ]
    },
    {
        "id": "P4",
        "title": "Sustainable Tech: Greener Data Centers",
        "keywords": [
            "energy efficiency", "liquid cooling", "renewable power",
            "carbon footprint", "PUE (Power Usage Effectiveness)"
        ]
    },
    {
        "id": "P5",
        "title": "Cybersecurity Trends for 2025",
        "keywords": [
            "zero trust", "AI threat detection", "ransomware",
            "IoT security", "data privacy"
        ]
    },
]

# 4) LIST OF CANDIDATE MODELS
models = list(pricing.keys())

# 5) OPENAI CALL
def call_openai(model_name, system_msg, user_msg):
    try:
        resp = openai.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
        )
        text = resp.output_text
        usage = {
            "prompt_tokens":    resp.usage.input_tokens,
            "completion_tokens":resp.usage.output_tokens,
            "total_tokens":     resp.usage.total_tokens
        }
        return text, usage
    except Exception as e:
        print(resp.json())
        print(f"Error calling OpenAI: {e}")
        return None, None

# 6) GEMINI CALL VIA SDK
def call_gemini(model_name, system_msg, user_msg, wpt):
    contents = [system_msg, user_msg]
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=2000,
            temperature=0.7
        )
    )
    text = response.text

    # approximate token counts
    words_in  = len(" ".join(contents).split())
    words_out = len(text.split())
    in_toks   = max(1, int(words_in  / wpt))
    out_toks  = max(1, int(words_out / wpt))

    usage = {
        "prompt_tokens":    in_toks,
        "completion_tokens":out_toks,
        "total_tokens":     in_toks + out_toks
    }
    return text, usage

# 7) COST CALCULATION
def compute_cost(model_name, in_toks, out_toks):
    in_rate, out_rate, _ = pricing[model_name]
    return (in_toks/1e6) * in_rate + (out_toks/1e6) * out_rate

# 8) MAIN PIPELINE
def main():
    system_instruction = (
        "You are an expert tech writer with a friendly, witty personality. "
        "Produce fact-checked, safe, and highly accurate articles."
    )

    # Load existing results if file exists
    existing_results = {}
    json_path = "raw_outputs/content_with_costs.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as fp:
            existing_results = json.load(fp)

    for model in models:
        print(f"\n=== Evaluating {model} ===")
        in_rate, out_rate, wpt = pricing[model]
        
        # Initialize model results if not exists
        if model not in existing_results:
            existing_results[model] = []

        for item in prompts:
            pid   = item["id"]
            title = item["title"]
            kws   = item["keywords"]

            user_instruction = (
                f"Title: \"{title}\"\n"
                "- Write a 1,500 - 2,500 word article.\n"
                "- Cite at least two reputable sources with URLs.\n"
                f"- Cover these keywords: {', '.join(kws)}.\n"
                "- Avoid unsafe, biased, or sensitive content.\n"
                "- Use a warm, conversational tone with a light joke or analogy."
            )

            start = time.time()
            if model.startswith("gemini"):
                text, usage = call_gemini(model, system_instruction, user_instruction, wpt)
            else:
                text, usage = call_openai(model, system_instruction, user_instruction)
            latency_ms = int((time.time() - start) * 1000)

            in_toks   = usage["prompt_tokens"]
            out_toks  = usage["completion_tokens"]
            cost_usd  = compute_cost(model, in_toks, out_toks)

            record = {
                "id":                pid,
                "title":             title,
                "keywords":          kws,
                "prompt_tokens":     in_toks,
                "completion_tokens": out_toks,
                "total_tokens":      in_toks + out_toks,
                "latency_ms":        latency_ms,
                "cost_usd":          round(cost_usd, 4),
                "response":          text
            }

            # Find if prompt ID exists in model's results
            found = False
            for i, existing_record in enumerate(existing_results[model]):
                if existing_record["id"] == pid:
                    existing_results[model][i] = record
                    found = True
                    break
            
            # If not found, append new record
            if not found:
                existing_results[model].append(record)

            print(f"  {pid}: lat={latency_ms}ms, in={in_toks} tok, out={out_toks} tok, cost=${cost_usd:.4f}")

            # Save after each prompt to ensure we don't lose data
            os.makedirs("raw_outputs", exist_ok=True)
            with open(json_path, "w") as fp:
                json.dump(existing_results, fp, indent=2)

    print(f"\n✅ Results saved to {json_path}")

if __name__ == "__main__":
    main()