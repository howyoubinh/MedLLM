import spacy
import scispacy
import re
from typing import List, Dict
from scispacy.linking import EntityLinker

class EntityExtractor:
    """A class to handle biomedical entity extraction using scispacy and UMLS."""
    
    def __init__(self):
        """Initialize the EntityExtractor with spacy model and UMLS linker."""
        self.nlp = spacy.load('en_core_sci_sm')
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.age_pattern = r'\b\d+[-\s]?(?:year|yr)s?[-\s]?old\b'

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using UMLS linking and custom age pattern matching.
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            List[Dict]: List of dictionaries containing entity information
        """
        if not text:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        # Process each entity found by spaCy/UMLS
        for ent in doc.ents:
            entity_info = {'text': ent.text, 'type': 'UNKNOWN'}
            
            # Check if there's a UMLS link
            if ent._.kb_ents and len(ent._.kb_ents) > 0:
                umls_id, score = ent._.kb_ents[0]
                linker = self.nlp.get_pipe("scispacy_linker")
                concept = linker.kb.cui_to_entity[umls_id]
                
                entity_info.update({
                    'umls_id': umls_id,
                    'score': score,
                    'semantic_types': concept.types,
                    'definition': concept.definition
                })
                
                if concept.types:
                    entity_info['type'] = concept.types[0]
        
            # Check if this entity matches our age pattern
            if re.search(self.age_pattern, ent.text, re.IGNORECASE):
                entity_info['type'] = 'AGE_VALUE'
            
            entities.append(entity_info)
        
        # Add any age patterns that weren't caught as entities
        text_spans = [(m.start(), m.end(), m.group()) for m in re.finditer(self.age_pattern, text, re.IGNORECASE)]
        for start, end, age_text in text_spans:
            if not any(age_text in e['text'] for e in entities):
                entities.append({
                    'text': age_text,
                    'type': 'AGE_VALUE',
                    'umls_id': None,
                    'score': None,
                    'semantic_types': None,
                    'definition': None
                })
        
        return entities

# For testing the module independently
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    # Test example
    test_text = "A 65-year-old female patient presents with persistent abdominal pain and fever."
    entities = extractor.extract_entities(test_text)
    
    print("\nTest Results:")
    print(f"Input text: {test_text}")
    print("\nExtracted entities:")
    for entity in entities:
        print(f"\nEntity: {entity['text']}")
        print(f"Type: {entity['type']}")
        if entity.get('umls_id'):
            print(f"UMLS ID: {entity['umls_id']}")
            print(f"Score: {entity['score']:.2f}")
            print(f"Semantic Types: {entity['semantic_types']}")
            print(f"Definition: {entity['definition']}")