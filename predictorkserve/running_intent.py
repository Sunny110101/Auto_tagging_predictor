import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from pathlib import Path
from typing import Dict
import logging
import joblib
from kserve import Model, ModelServer
import subprocess

class IntentPredictorModel(Model):
    def __init__(self, name: str, model_file: str, config_file: str, label_mapping_path: str, output_path: str = '/tmp/outputs'):
        super().__init__(name)
        self.name = name
        self.model_file = Path(model_file)
        self.config_file = Path(config_file)
        self.label_mapping_path = label_mapping_path
        self.output_path = Path(output_path)
        self.ready = False
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing model from local path: {self.model_file}")
        self.load_model_artifacts()

    def save_inference_url(self):
        """Save the inference URL for the model"""
        try:
            namespace = os.getenv('NAMESPACE', 'kubeflow-user-example-com')
            
            # Create output directories
            output_dirs = [
                self.output_path / 'InferenceService_Status',
                Path('/pipeline/output'),
                Path('/pipeline/outputs')
            ]
            
            for dir_path in output_dirs:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created output directory: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"Could not create directory {dir_path}: {e}")

            # Get inference URL
            cmd = f"kubectl get inferenceservice {self.name} -n {namespace} -o jsonpath='{{.status.url}}'"
            self.logger.info(f"Executing command: {cmd}")
            
            try:
                url = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
                self.logger.info(f"Retrieved URL: {url}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error getting URL: {e.output.decode() if e.output else str(e)}")
                url = "URL_NOT_AVAILABLE"

            # Save to all output directories
            for output_dir in output_dirs:
                try:
                    # Save detailed info
                    info_file = output_dir / "inference_service_info.json"
                    info = {
                        "inference_url": f"{url}/v1/models/{self.name}:predict",
                        "model_name": self.name,
                        "namespace": namespace,
                        "timestamp": datetime.now().isoformat(),
                        "example_payload": {
                            "instances": [{"text": "example text"}]
                        }
                    }
                    
                    with info_file.open('w') as f:
                        json.dump(info, f, indent=2)
                    self.logger.info(f"Saved inference info to: {info_file}")
                    
                    # Save URL only
                    url_file = output_dir / "inference_url.txt"
                    with url_file.open('w') as f:
                        f.write(f"{url}/v1/models/{self.name}:predict")
                    self.logger.info(f"Saved URL to: {url_file}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not save to {output_dir}: {e}")

        except Exception as e:
            self.logger.error(f"Error in save_inference_url: {str(e)}")
            self.logger.exception("Detailed error traceback:")


    def load_model_artifacts(self):
        """Load all required model artifacts"""
        try:
            # Load model configuration
            self.logger.info(f"Loading config from: {self.config_file}")
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)

            # Load intent mapping
            self.load_intent_mapping()
            self.ready = True

            # Get BERT model path
            bert_path = os.path.join(os.environ.get('APP', '/app'), 'bert-base-uncased')
            self.logger.info(f"Looking for BERT model at: {bert_path}")
            self.save_inference_url()
            
            if not os.path.exists(bert_path):
                self.logger.error(f"BERT model path does not exist: {bert_path}")
                raise FileNotFoundError(f"BERT model directory not found at {bert_path}")

            # Initialize tokenizer and model from local path
            self.logger.info("Initializing tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
            
            self.logger.info("Initializing base model")
            base_model = AutoModel.from_pretrained(bert_path, local_files_only=True)

            # Initialize model architecture
            self.logger.info("Creating model architecture")
            self.model = self.create_model_architecture(base_model)

            # Load saved model state
            self.logger.info("Loading model state")
            self.load_model_state()

            # Set device and model mode
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {self.device}")
            self.model.to(self.device)
            self.model.eval()

            self.ready = True
            self.logger.info("Model initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {str(e)}")
            self.logger.exception("Detailed error traceback:")
            raise

    def create_model_architecture(self, base_model):
        """Create the model architecture"""
        class IntentClassifier(torch.nn.Module):
            def __init__(self, base_model, hidden_size, num_labels, dropout):
                super().__init__()
                self.base_model = base_model
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(base_model.config.hidden_size, hidden_size),
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.LayerNorm(hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size // 2, num_labels)
                )

            def forward(self, input_ids, attention_mask):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                attention_mask = attention_mask.unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                pooled_output = torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
                return self.classifier(pooled_output)

        return IntentClassifier(
            base_model=base_model,
            hidden_size=self.config['hidden_size'],
            num_labels=self.config['num_labels'],
            dropout=self.config['dropout']
        )

    def load_intent_mapping(self):
        """Load intent mapping from file or config"""
        try:
            self.logger.info(f"Loading intent mapping from: {self.label_mapping_path}")
            if self.label_mapping_path:
                with open(self.label_mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.intent_mapping = {
                        'intent_to_id': {str(k): str(v) for k, v in mapping_data['intent_to_id'].items()},
                        'id_to_intent': {str(k): str(v) for k, v in mapping_data['id_to_intent'].items()}
                    }
            else:
                self.logger.info("Using default intent mapping from config")
                self.intent_mapping = {
                    'id_to_intent': {str(k): v for k, v in self.config['id2label'].items()},
                    'intent_to_id': {v: str(k) for k, v in self.config['id2label'].items()}
                }
            self.logger.info("Intent mapping loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading intent mapping: {str(e)}")
            raise

    def load_model_state(self):
        """Load the model state using joblib"""
        try:
            self.logger.info(f"Loading model state from: {self.model_file}")
            model_state = joblib.load(self.model_file)

            if isinstance(model_state, dict) and 'state_dict' in model_state:
                state_dict = model_state['state_dict']
            else:
                state_dict = model_state

            # Remove problematic keys
            if 'base_model.embeddings.position_ids' in state_dict:
                self.logger.info("Removing position_ids from state dict")
                del state_dict['base_model.embeddings.position_ids']

            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info("Model state loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model state: {str(e)}")
            raise

    def predict(self, request: Dict) -> Dict:
        """Request format:
        {
            "instances": [{"text": "your query here"}, ...]
        }
        """
        try:
            instances = request["instances"]
            texts = [instance.get("text", "") for instance in instances]

            predictions = []
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                )

                # Move to device
                model_inputs = {
                    'input_ids': inputs['input_ids'].to(self.device),
                    'attention_mask': inputs['attention_mask'].to(self.device)
                }

                # Get predictions
                with torch.no_grad():
                    logits = self.model(**model_inputs)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)

                # Process results
                pred_idx = torch.argmax(probabilities, dim=1)[0]
                confidence = float(probabilities[0][pred_idx])
                pred_idx_str = str(int(pred_idx))
                predicted_intent = self.intent_mapping['id_to_intent'].get(
                    pred_idx_str,
                    f"unknown_intent_{pred_idx_str}"
                )

                # Get all probabilities
                intent_probs = {}
                for i, prob in enumerate(probabilities[0]):
                    idx_str = str(i)
                    intent_name = self.intent_mapping['id_to_intent'].get(idx_str, f"Intent_{idx_str}")
                    intent_probs[intent_name] = float(prob)

                predictions.append({
                    'input_text': text,
                    'prediction': {
                        'intent': predicted_intent,
                        'confidence': confidence,
                        'intent_id': int(pred_idx)
                    },
                    'all_probabilities': intent_probs,
                    'timestamp': datetime.now().isoformat()
                })

            return {"predictions": predictions}

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description="Intent Prediction Model")
    parser.add_argument('--model_file', type=str, required=True, help="Path to the trained model file")
    parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    parser.add_argument('--label_mapping_path', type=str, required=True, help="Path to the label mapping file")
    parser.add_argument('--output_path', type=str, default='/tmp/outputs', help="Path to save the inference URL")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        model = IntentPredictorModel(
            name="intent-predictor",
            model_file=args.model_file,
            config_file=args.config_file,
            label_mapping_path=args.label_mapping_path,
            output_path=args.output_path
        )
        logging.info("Model initialized and URL saved successfully")
        
        # Print all saved locations
        logging.info("Check the following locations for the inference URL:")
        for path in ['/pipeline/output', '/pipeline/outputs', '/tmp/outputs/InferenceService_Status']:
            if os.path.exists(path):
                logging.info(f"- {path}")
                
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()