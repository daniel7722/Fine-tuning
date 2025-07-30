"""
The synthetic data for Emergency Event Detection (fake). 

Objective: 
  Each scene (an "episode") is labeled as 0 (normal) or 1 (emergency).
  Agents are sensors that observe different modalities and emit predictions. The fusion unit learns to inferemergencies by weighting each modality dynamically, based on reliability and history.

Scene Configuration: 
  - Weather: foggy, rainy, clear --> foggy impairs vision
  - Lighting: dat, night, between --> night help IR but impairs vision
  - Temperature: cold, mild, hot --> cold might help identify body heat
  - Audio Clarity: low, medium, high --> low clarity impairs audio sensor
  - Unusual Sound: true, false --> true indicates potential emergency
  - Heat Signature: true, false (e.g. person lying down) --> strong emergency signal
  - Motion Pattern: normal, erratic, none --> erratic motion can signal distress

Agent Configuration: 
Modalities: 
  - Vision (0): motion_pattern, weather, lighting
  - Audio (1): unusual_sound, audio_clarity
  - Infrared (2): heat_signature, temperature, lighting

Each agent sees only what its modality supports, and its correctness depends on signal quality + logic. 


Example Scene: 
{
  "weather": "foggy",
  "lighting": "night",
  "temperature": "cold",
  "audio_clarity": "low",
  "unusual_sound": true,
  "heat_signature": true,
  "motion_pattern": "erratic",
  "label": 1 # Ground truth
}
--> Audio sensor might produce less reliable data due to low clarity, but IR sensor detects heat signal coupled with erratic motion, leading to a positive emergency label.

This will be saved in a .jsonl file. 
"""
import json
import random
from pathlib import Path

def generate_scene(): 
    lighting = random.choices(["day", "night", "between"], weights=[0.5, 0.3, 0.2])[0]
    weather = random.choices(["foggy", "rainy", "clear"], weights=[0.5, 0.3, 0.2])[0]
    temperature = (
        "cold" if lighting == "night" and weather == "foggy" and random.random() > 0.33
        else "hot" if lighting == "day" and weather == "clear" and random.random() > 0.33
        else "mild"
    )
    audio_clarity = random.choices(["low", "medium", "high"], weights=[0.5, 0.3, 0.2])[0]
    unusual_sound = random.random() < 0.1
    heat_signature = random.random() < 0.2
    motion_pattern = random.choices(["normal", "erratic", "none"], weights=[0.6, 0.15, 0.25])[0]

    return {
        "weather": weather,
        "lighting": lighting,
        "temperature": temperature,
        "audio_clarity": audio_clarity,
        "unusual_sound": unusual_sound,
        "heat_signature": heat_signature,
        "motion_pattern": motion_pattern
    }

def generate_ground_truth(scene): 
    score = 0
    if scene["unusual_sound"]: score += 2
    if scene["heat_signature"]: score += 1.5
    if scene["motion_pattern"] == "erratic": score += 1.5
    if scene["motion_pattern"] == "none" and scene["heat_signature"]: score += 1.0
    if scene["unusual_sound"] and scene["audio_clarity"] == "low": score -= 0.5
    if scene["weather"] == "foggy" and scene["motion_pattern"] == "erratic": score -= 0.5

    return 1 if score >= 2.5 else 0

def generate_episode(): 
    scene = generate_scene()
    scene["label"] = generate_ground_truth(scene)
    return scene

if __name__ == "__main__":
    output_path = Path("data/synthetic/emergency_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emergency_data = []
    non_emergency_data = []
    while len(emergency_data) < 5000 or len(non_emergency_data) < 5000: 
        episode = generate_episode()
        if episode["label"] == 1:
            emergency_data.append(episode)
        else:
            non_emergency_data.append(episode)
    # Balance the dataset
    min_count = min(len(emergency_data), len(non_emergency_data))
    print("min_count:", min_count)
    balanced = emergency_data[:min_count] + non_emergency_data[:min_count]
    random.shuffle(balanced)
    with open(output_path, "w") as f: 
        for episode in balanced:
            f.write(json.dumps(episode) + "\n")