import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration


# Initialize model configuration
class CustomBartModel(nn.Module):
    def __init__(self, config_path=None):
        super(CustomBartModel, self).__init__()
        if config_path:
            self.config = BartConfig.from_pretrained(config_path)
        else:
            self.config = BartConfig(
                d_model=768,
                encoder_layers=6,
                decoder_layers=6,
                encoder_attention_heads=8,
                decoder_attention_heads=8,
                scale_embedding=True,
                vocab_size=50265
            )
        self.model = BartForConditionalGeneration(self.config)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        # Forward pass
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

# Initialize model for distributed training
def initialize_model(config_path=None, device=None):
    model = CustomBartModel(config_path=config_path)
    
    # Move model to the correct device
    if device:
        model.to(device)
    return model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = initialize_model(device=device)
# # Example input data
# input_ids = torch.tensor([[0, 1, 2, 3]])  # Tokenized input sequence
# decoder_input_ids = torch.tensor([[0, 4, 5]])
