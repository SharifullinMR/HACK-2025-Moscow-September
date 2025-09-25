import re
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    pipeline
)
from typing import List, Tuple, Dict, Any

class NERModel:
    def __init__(self, model_path: str = "./ner-model/checkpoint-18396"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.label2id = None
        self.id2label = None
        os.path.abspath(model_path)
        self._load_model()
    
    def _load_model(self):
        "Загрузка модели"
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Путь к модели не существует: {self.model_path}")
            
            print("Загрузка токенизатора...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            print("Загрузка модели...")
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            
            print("pipeline")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            
            self.label2id = {
                'B-BRAND': 0, 'B-PERCENT': 1, 'B-TYPE': 2, 'B-VOLUME': 3,
                'I-BRAND': 4, 'I-TYPE': 5, 'O': 6
            }
            self.id2label = {v: k for k, v in self.label2id.items()}
            
            print("Модель загружена")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            raise Exception(f"Ошибка загрузки модели: {str(e)}")
    
    def predict_bio(self, query: str) -> List[Tuple[int, int, str]]:
        "Предсказание"
        if not query or not query.strip():
            return []
        
        try:
         
            predictions = self.ner_pipeline(query)
            
      
            predictions = sorted(predictions, key=lambda x: x['start'])

            words = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', query)]

            word_labels = []
            prev_entity_type = None
            
            for w_start, w_end, word in words:

                tokens_in_word = [t for t in predictions if t['start'] >= w_start and t['end'] <= w_end]
                
                if not tokens_in_word:
                    word_labels.append((w_start, w_end, "O"))
                    prev_entity_type = None
                    continue

                first_entity = tokens_in_word[0]['entity_group']
          
                base_entity = first_entity[2:] if first_entity.startswith(("B-", "I-")) else first_entity

                if prev_entity_type == base_entity:
                    bio_label = f"I-{base_entity}"
                else:
                    bio_label = f"B-{base_entity}"
                    prev_entity_type = base_entity

                word_labels.append((w_start, w_end, bio_label))

            return word_labels
            
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def batch_predict(self, queries: List[str]) -> List[List[Tuple[int, int, str]]]:
        return [self.predict_bio(query) for query in queries]



ner_model = None

def initialize_model():
    global ner_model
    try:
       
        model_path = r"C:\Users\Marsohodik\Downloads\hack\pipeline\ner-model\checkpoint-18396"
        
        
        if not os.path.exists(model_path):
            print(f"Директория не существует: {model_path}")
            # Пробуем относительный путь
            model_path = "./ner-model/checkpoint-18396"
            print({os.path.abspath(model_path)})
        
        ner_model = NERModel(model_path)
        print("Модель инициализирована")
        return ner_model
    except Exception as e:
        print(f"Ошибка{e}")
        return None


def predict_dataframe(df, text_column='sample'):
   
    if ner_model is None:
        raise Exception("Модель не инициализирована")
    
    annotations = []
    for idx, row in df.iterrows():
        ann = ner_model.predict_bio(row[text_column])
        annotations.append(ann)
    
    df_result = df.copy()
    df_result['annotation'] = annotations
    return df_result


if __name__ == "__main__":
    
    initialize_model()
    
    pass