import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class GeneralDQN(pl.LightningModule):
    def __init__(self, model, input_dim, vector_dim, action_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model  # Generic model (CNN, MLP, etc.)
        
        # Compute model output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros((1, *input_dim))  # (batch, channels, H, W) or (batch, features)
            model_out_dim = self.model(dummy_input).shape[1]
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(model_out_dim + vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Q-values for each action
        )
        
    def forward(self, state, vectors):
        features = self.model(state)  # Process input through the given model
        x = torch.cat([features, vectors], dim=1)  # Concatenate with positional vectors
        q_values = self.fc(x)  # Get Q-values
        return q_values
    
    def training_step(self, batch, batch_idx):
        state, vectors, action, reward, next_state, done = batch
        q_values = self(state, vectors)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self(next_state, vectors).max(1)[0]
            target_q_value = reward + (1 - done.float()) * 0.99 * next_q_values
        
        loss = nn.functional.mse_loss(q_value, target_q_value)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
