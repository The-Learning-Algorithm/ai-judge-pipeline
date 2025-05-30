# AI Judge Pipeline

![alt text](/images/ai_judge_pipeline.png)

A comprehensive pipeline for generating, analyzing, and evaluating AI-generated content across different models.

## Project Structure

The pipeline consists of four main scripts:

1. `1_generate_content.py`: Generates content using different AI models
2. `2_ai_judge.py`: Analyzes content for word count and broken links
3. `3_ai_judge.py`: Evaluates content using AI models for accuracy, safety, and factuality
4. `4_find_the_winner.py`: Determines the best performing model based on weighted metrics

## Setup

1. Clone the repository:
```bash
git clone https://github.com/The-Learning-Algorithm/ai-judge-pipeline?tab=readme-ov-file
cd ai-judge-pipeline
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file with your API keys
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

Run the pipeline in sequence:

1. Generate content:
```bash
python 1_generate_content.py
```

2. Analyze content:
```bash
python 2_ai_judge.py
```

3. Evaluate content:
```bash
python 3_ai_judge.py
```

4. Find the winner:
```bash
python 4_find_the_winner.py
```

## Output Files

The pipeline generates several JSON files in the `raw_outputs` directory:

- `content_with_costs.json`: Generated content with cost information
- `content_with_analysis.json`: Content with word count and link analysis
- `content_with_judgment.json`: Content with AI evaluation metrics
- `contest_results.json`: Final results with model rankings

## Evaluation Metrics

The final evaluation uses weighted metrics:

- Cost: 25%
- Accuracy: 30%
- Factuality: 15%
- Safety: 10%
- Word Count: 10%
- Latency: 10%

## Project Structure

```
ai-judge-pipeline/
├── 1_generate_content.py
├── 2_ai_judge.py
├── 3_ai_judge.py
├── 4_find_the_winner.py
├── requirements.txt
├── .env
├── .gitignore
└── raw_outputs/
    ├── content_with_costs.json
    ├── content_with_analysis.json
    ├── content_with_judgment.json
    └── contest_results.json
```

## Dependencies

- Python 3.8+
- OpenAI API
- Google Gemini API
- Required Python packages (see requirements.txt)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact
Amir Tadrisi 
Site: [The Learning Algorithm](https://thelearningalgorithm.ai)
Email: [amirtds@gmail.com](mailto:amirtds@gmail.com)