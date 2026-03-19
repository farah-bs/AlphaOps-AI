"""
LSTMDirectionModel — AlphaOps AI
==================================
Modèle LSTM pour la classification de direction multi-horizon.

Prédit la probabilité que le prix soit en hausse à J+1, J+7 et J+30
à partir d'une séquence de plusieurs jours de features techniques.

Architecture (schématique) :
    Input  : (batch, seq_len, input_size)
    LSTM   : hidden=64, layers=2, dropout=0.2
    Attention : weighted sum sur tous les timesteps
    Dense  : 64 → 32 (ReLU) → 3 logits
    Output : (batch, 3)  — logits pour [J+1, J+7, J+30]

Note : pas de sigmoid en sortie — utiliser BCEWithLogitsLoss à l'entraînement
et torch.sigmoid() à l'inférence.
"""

# Import lazy pour ne pas forcer torch dans les modules qui n'en ont pas besoin
def _get_model_class():
    import torch
    import torch.nn as nn

    class LSTMDirectionModel(nn.Module):
        def __init__(
            self,
            input_size:  int   = 12,
            hidden_size: int   = 64,
            num_layers:  int   = 2,
            dropout:     float = 0.2,
            n_outputs:   int   = 3,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attn = nn.Linear(hidden_size, 1)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_outputs),
            )

        def forward(self, x):
            # x : (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # attention temporelle : apprend quels jours sont pertinents
            attn_w = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq, 1)
            last = (lstm_out * attn_w).sum(dim=1)               # (batch, hidden)
            return self.head(last)

    return LSTMDirectionModel


def build_model(input_size=12, hidden_size=64, num_layers=2, dropout=0.2, n_outputs=3):
    """Instancie LSTMDirectionModel avec les hyperparamètres donnés."""
    LSTMDirectionModel = _get_model_class()
    return LSTMDirectionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_outputs=n_outputs,
    )
