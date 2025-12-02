"""
Simple script to load and explore the dataset
"""
import json
import pickle

def load_json():
    """Load human-readable JSON version"""
    with open('complete_dataset.json', 'r') as f:
        return json.load(f)

def load_pickle():
    """Load Python pickle version (faster)"""
    with open('complete_dataset.pkl', 'rb') as f:
        return pickle.load(f)

def print_summary(data):
    """Print dataset summary"""
    print(f"Total data points: {len(data)}")
    print(f"\nFirst data point example:")
    print(json.dumps(data[0], indent=2))
    
    # Count by scenario
    scenarios = {}
    for d in data:
        sid = d['track_id']
        scenarios[sid] = scenarios.get(sid, 0) + 1
    
    print(f"\nData points per scenario:")
    for sid in sorted(scenarios.keys()):
        print(f"  Scenario {sid}: {scenarios[sid]} points")

if __name__ == "__main__":
    print("Loading dataset...")
    data = load_json()  # or use load_pickle()
    print_summary(data)
    
    print("\nData is now loaded in 'data' variable")
    print("Access it like: data[0]['travel_time']")
