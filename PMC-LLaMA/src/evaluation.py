import warnings
from src.classes.entity_extractor import EntityExtractor
# Ignore spacy warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


no_entities=True

#### ENTITY EXTRACTION ####
# Set up entity extractor
extractor = None
if no_entities:
    print(f"No Entity Extractor")
    extractor = None
else:
    print(f"\033[32mInitializing Entity Extractor\033[0m")
    extractor = EntityExtractor()