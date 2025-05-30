import json
import numpy as np

def normalize_value(value, min_val, max_val):
    """Normalize a value between 0 and 1."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def find_bounds(data):
    """Find min and max values for each metric across all models and prompts."""
    bounds = {
        'latency_ms': {'min': float('inf'), 'max': float('-inf')},
        'cost_usd': {'min': float('inf'), 'max': float('-inf')},
        'words_count': {'min': float('inf'), 'max': float('-inf')},
        'accuracy': {'min': float('inf'), 'max': float('-inf')},
        'safety': {'min': float('inf'), 'max': float('-inf')},
        'factuality': {'min': float('inf'), 'max': float('-inf')}
    }
    
    for model_results in data.values():
        for result in model_results:
            for metric in bounds:
                value = result[metric]
                bounds[metric]['min'] = min(bounds[metric]['min'], value)
                bounds[metric]['max'] = max(bounds[metric]['max'], value)
    
    return bounds

def calculate_model_score(model_results, bounds, weights):
    """Calculate weighted score for a model based on its results."""
    normalized_scores = []
    
    for result in model_results:
        # Normalize each metric (inverse for latency and cost since lower is better)
        latency_norm = 1 - normalize_value(result['latency_ms'], bounds['latency_ms']['min'], bounds['latency_ms']['max'])
        cost_norm = 1 - normalize_value(result['cost_usd'], bounds['cost_usd']['min'], bounds['cost_usd']['max'])
        words_norm = normalize_value(result['words_count'], bounds['words_count']['min'], bounds['words_count']['max'])
        accuracy_norm = normalize_value(result['accuracy'], bounds['accuracy']['min'], bounds['accuracy']['max'])
        safety_norm = normalize_value(result['safety'], bounds['safety']['min'], bounds['safety']['max'])
        factuality_norm = normalize_value(result['factuality'], bounds['factuality']['min'], bounds['factuality']['max'])
        
        # Calculate weighted score
        score = (
            weights['cost'] * cost_norm +
            weights['latency'] * latency_norm +
            weights['word_count'] * words_norm +
            weights['accuracy'] * accuracy_norm +
            weights['safety'] * safety_norm +
            weights['factuality'] * factuality_norm
        )
        normalized_scores.append(score)
    
    return np.mean(normalized_scores)

def find_winner():
    # Load judgment results
    with open('raw_outputs/content_with_judgment.json', 'r') as f:
        data = json.load(f)
    
    # Define weights
    weights = {
        'cost': 0.25,
        'latency': 0.10,
        'word_count': 0.10,
        'accuracy': 0.30,
        'safety': 0.10,
        'factuality': 0.15
    }
    
    # Find bounds for normalization
    bounds = find_bounds(data)
    
    # Calculate scores for each model
    model_scores = {}
    for model, results in data.items():
        score = calculate_model_score(results, bounds, weights)
        model_scores[model] = score
    
    # Find winner
    winner = max(model_scores.items(), key=lambda x: x[1])
    
    # Prepare results
    results = {
        'bounds': bounds,
        'model_scores': model_scores,
        'winner': {
            'model': winner[0],
            'score': winner[1]
        },
        'weights': weights
    }
    
    # Save results
    with open('raw_outputs/contest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n=== Contest Results ===")
    print("\nBounds for each metric:")
    for metric, values in bounds.items():
        print(f"{metric}: min={values['min']:.2f}, max={values['max']:.2f}")
    
    print("\nModel Scores:")
    for model, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {score:.4f}")
    
    print(f"\n🏆 Winner: {winner[0]} with score {winner[1]:.4f}")
    print("\nResults saved to raw_outputs/contest_results.json")

if __name__ == "__main__":
    find_winner() 