import json
from pathlib import Path
from augmentations import TextAugmentor
from typing import List, Dict
import argparse
from tqdm import tqdm
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser(description='Generate augmented questions using Llama 3.2')
    parser.add_argument('--input-file', type=str, default='test_4_options.jsonl',
                        help='Input JSONL file containing original questions')
    parser.add_argument('--output-file', type=str, default='augmented_questions.jsonl',
                        help='Output JSONL file for augmented questions')
    parser.add_argument('--augmentations', nargs='+', 
                       default=['synonyms', 'paraphrase', 'shuffle', 'expand'],
                       help='List of augmentation methods to apply')
    parser.add_argument('--model-name', type=str, default='llama3.3', # llama3.3 is way better
                        help='Model name for text augmentation')
    return parser.parse_args()

def load_questions(filepath: str) -> List[Dict]:
    questions = []
    with jsonlines.open(filepath) as reader:
        for item in reader:
            # questions.append(item)
            questions.append({'question': item['question']})
    return questions

def clean_entry(entry: Dict, augmentation_type: str, question_num: int) -> Dict:
    """Remove unnecessary fields and rename question field"""
    cleaned = {
        f"{question_num}{augmentation_type}": entry['question']
    }
    return cleaned

def augment_question(augmentor: TextAugmentor, question: Dict, methods: List[str], question_num: int) -> List[Dict]:
    augmented_entries = []
    
    # Add original question
    original_entry = clean_entry(question, 'original', question_num)
    augmented_entries.append(original_entry)
    
    # Generate augmentations
    original_text = question['question']
    for method in methods:
        try:
            augmented_text = getattr(augmentor, method)(original_text)
            
            # Create new entry with augmented text
            augmented_entry = question.copy()
            augmented_entry['question'] = augmented_text
            augmented_entry = clean_entry(augmented_entry, method, question_num)
            augmented_entries.append(augmented_entry)
            
        except Exception as e:
            print(f"Error applying {method} augmentation: {str(e)}")
            continue
            
    return augmented_entries

def main():
    args = parse_args()
    
    # Initialize augmentor
    print(f"Initializing text augmentor... using {args.model_name}")
    augmentor = TextAugmentor(model_name=args.model_name)
    
    # Load original questions
    print(f"Loading questions from {args.input_file}...")
    questions = load_questions(args.input_file)
    
    # Process each question
    print("Generating augmentations...")
    all_entries = []
    for i, question in enumerate(tqdm(questions), 1):
        augmented_entries = augment_question(augmentor, question, args.augmentations, i)
        all_entries.extend(augmented_entries)
    
    # Save results
    print(f"Saving augmented questions to {args.output_file}...")
    with jsonlines.open(args.output_file, mode='w') as writer:
        for entry in all_entries:
            writer.write(entry)
    
    print(f"Completed! Generated {len(all_entries)} total entries")
    print(f"Original questions: {len(questions)}")
    print(f"Augmented questions: {len(all_entries) - len(questions)}")

if __name__ == "__main__":
    main()
