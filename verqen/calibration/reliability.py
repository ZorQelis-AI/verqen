import numpy as np

def reliability_diagram_data(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> dict:
    """Returns bin edges, avg confidence, empirical accuracy, count, gap per bin."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    data = {
        'bin_edges': bin_edges.tolist(),
        'avg_confidence': [],
        'empirical_accuracy': [],
        'bin_count': [],
        'gap': []
    }
    
    for i in range(num_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == num_bins - 1:
            bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        
        count = np.sum(bin_mask)
        if count > 0:
            avg_conf = np.mean(confidences[bin_mask])
            emp_acc = np.mean(accuracies[bin_mask])
            data['avg_confidence'].append(float(avg_conf))
            data['empirical_accuracy'].append(float(emp_acc))
            data['bin_count'].append(int(count))
            data['gap'].append(float(np.abs(emp_acc - avg_conf)))
        else:
            data['avg_confidence'].append(0.0)
            data['empirical_accuracy'].append(0.0)
            data['bin_count'].append(0)
            data['gap'].append(0.0)
            
    return data

def reliability_report(predictions: list[dict], num_bins: int = 10) -> dict:
    """Full reliability report broken down by type, domain, difficulty."""
    # Assuming predictions contain 'confidence', 'correct', 'type', 'domain', 'difficulty'
    types = set(p.get('type', 'unknown') for p in predictions)
    domains = set(p.get('domain', 'unknown') for p in predictions)
    difficulties = set(p.get('difficulty', 'unknown') for p in predictions)
    
    report = {
        'overall': {},
        'by_type': {},
        'by_domain': {},
        'by_difficulty': {}
    }
    
    def process_subset(subset: list[dict]) -> dict:
        if not subset:
            return {}
        confidences = np.array([p['confidence'] for p in subset])
        accuracies = np.array([float(p['correct']) for p in subset])
        return reliability_diagram_data(confidences, accuracies, num_bins)
    
    report['overall'] = process_subset(predictions)
    
    for t in types:
        report['by_type'][t] = process_subset([p for p in predictions if p.get('type', 'unknown') == t])
        
    for d in domains:
        report['by_domain'][d] = process_subset([p for p in predictions if p.get('domain', 'unknown') == d])
        
    for d in difficulties:
        report['by_difficulty'][d] = process_subset([p for p in predictions if p.get('difficulty', 'unknown') == d])
        
    return report
