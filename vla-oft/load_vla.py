# This script loads the OpenVLA-OFT model from a given huggingface id or path and prepares it for inference.

# import torch


class LoadVLA:
    def __init__(self, model_id_or_path):
        self.model_id_or_path = model_id_or_path

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)

    def prepare_model_for_inference(self):
        self.model.eval()
        self.model.to(torch.bfloat16)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_model_id_or_path(self):
        return self.model_id_or_path

    def get_model_type(self):
        return self.model_type

class OpenVLAOFT(LoadVLA):
    def __init__(self, model_id_or_path):
        super().__init__(model_id_or_path)
        self.model_type = "openvla-oft"

    def load_model(self):
        super().load_model()
        self.model.eval()
        self.model.to(torch.bfloat16)

    def prepare_model_for_inference(self):
        super().prepare_model_for_inference()
