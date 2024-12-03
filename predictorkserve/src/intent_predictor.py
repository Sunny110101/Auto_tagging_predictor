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
from kserve import Model, ModelServer, KServeClient
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1ModelFormat
from kserve import constants
from kserve import V1beta1ModelSpec
from kubernetes import client
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
        self.create_inference_service()

    def create_inference_service(self):
        try:
            kserve_client = KServeClient()
            namespace = os.getenv('NAMESPACE', 'kserve-test')
            
            # Convert paths to absolute paths
            model_path = os.path.abspath(self.model_file)
            config_path = os.path.abspath(self.config_file)
            mapping_path = os.path.abspath(self.label_mapping_path)
            
            self.logger.info(f"Using model path: {model_path}")
            self.logger.info(f"Using config path: {config_path}")
            self.logger.info(f"Using mapping path: {mapping_path}")

            isvc = V1beta1InferenceService(
                api_version=constants.KSERVE_V1BETA1,
                kind=constants.KSERVE_KIND,
                metadata=client.V1ObjectMeta(
                    name=self.name,
                    namespace=namespace
                ),
                spec=V1beta1InferenceServiceSpec(
                    predictor=V1beta1PredictorSpec(
                        model=V1beta1ModelSpec(
                            model_format=V1beta1ModelFormat(
                                name='pytorch'
                            ),
                            resources=client.V1ResourceRequirements(
                                requests={
                                    'cpu': '100m',
                                    'memory': '1Gi'
                                },
                                limits={
                                    'cpu': '4',
                                    'memory': '16Gi'
                                }
                            ),
                            runtime='kserve-pytorch',
                            protocol_version='v2',
                            storage_uri=f'file://{os.path.join(self.model_file, "model.joblib")}',
                            env=[
                                client.V1EnvVar(
                                    name="MODEL_PATH",
                                    value=model_path
                                ),
                                client.V1EnvVar(
                                    name="CONFIG_PATH",
                                    value=config_path
                                ),
                                client.V1EnvVar(
                                    name="MAPPING_PATH",
                                    value=mapping_path
                                )
                            ]
                        )
                    )
                )
            )

            # Create/update service
            self.logger.info(f"Creating InferenceService {self.name}")
            kserve_client.create(isvc)
            
            self.logger.info("Waiting for InferenceService to be ready...")
            kserve_client.wait_isvc_ready(
                name=self.name,
                namespace=namespace,
                timeout_seconds=300
            )
            
            self.logger.info("InferenceService created successfully")
            self.save_inference_url()

        except Exception as e:
            self.logger.error(f"Error creating InferenceService: {str(e)}")
            self.logger.error(f"Model path exists: {os.path.exists(model_path)}")
            self.logger.error(f"Config path exists: {os.path.exists(config_path)}")
            self.logger.error(f"Mapping path exists: {os.path.exists(mapping_path)}")
            try:
                status = kserve_client.get(
                    name=self.name,
                    namespace=namespace
                )
                self.logger.error(f"Current service status: {status}")
            except Exception as status_error:
                self.logger.error(f"Could not get service status: {str(status_error)}")
            raise

    def save_inference_url(self):
        """Save the inference URL for the model"""
        try:
            namespace = os.getenv('NAMESPACE', 'kserve-test')
            
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

            # Get inference URL using KServe client
            kserve_client = KServeClient()
            isvc = kserve_client.get(
                name=self.name,
                namespace=namespace
            )
            url = isvc['status']['url'] if isvc.get('status', {}).get('url') else "URL_NOT_AVAILABLE"

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
        """Load intent mapping from file"""
        try:
            self.logger.info(f"Loading intent mapping from: {self.label_mapping_path}")
            with open(self.label_mapping_path, 'r') as f:
                mapping_data = json.load(f)
                self.intent_mapping = {
                    'intent_to_id': {str(k): str(v) for k, v in mapping_data['intent_to_id'].items()},
                    'id_to_intent': {str(k): str(v) for k, v in mapping_data['id_to_intent'].items()}
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
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description="Intent Prediction Model")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")
    parser.add_argument('--model_file', type=str, required=True, help="Path to the trained model file")
    parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    parser.add_argument('--label_mapping_path', type=str, required=True, help="Path to the label mapping file")
    parser.add_argument('--output_path', type=str, default='/tmp/outputs', help="Path to save the inference URL")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        model = IntentPredictorModel(
            name=args.model_name,
            model_file=args.model_file,
            config_file=args.config_file,
            label_mapping_path=args.label_mapping_path,
            output_path=args.output_path
        )
        ModelServer().start([model])
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()