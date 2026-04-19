import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# CONFIG
# ======================
BOARD_ROWS = 6
BOARD_COLS = 7
BOARD_CHANNELS = 2
NUM_ACTIONS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# BOARD ENCODING
# ======================
def encode_board_cnn(board):
    """
    Encodes a raw board (6x7) into a (2, 6, 7) tensor for the CNN.
    board values: 1 = current player, -1 = opponent, 0 = empty
    """
    encoded = np.zeros((BOARD_ROWS, BOARD_COLS, 2), dtype=np.float32)
    encoded[:, :, 0] = (board == 1)
    encoded[:, :, 1] = (board == -1)
    tensor = torch.tensor(encoded, dtype=torch.float32).permute(2, 0, 1)  # (2, 6, 7)
    return tensor.unsqueeze(0).to(DEVICE)  # (1, 2, 6, 7)

def extract_patches_transformer(board):
    """
    Extracts patches from a (6, 7, 2) board for the Transformer.
    Returns tensor of shape (1, 42, 2) — 42 patches, each of size 2.
    """
    PATCH_SIZE = 1
    STRIDE = 1
    patches = []
    for row in range(0, BOARD_ROWS - PATCH_SIZE + 1, STRIDE):
        for col in range(0, BOARD_COLS - PATCH_SIZE + 1, STRIDE):
            patch = board[row:row+PATCH_SIZE, col:col+PATCH_SIZE, :]
            patches.append(patch.ravel())
    patches = np.array(patches, dtype=np.float32)  # (42, 2)
    tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 42, 2)
    return tensor

def encode_board_transformer(board):
    """
    Encodes a raw board (6x7) into patch format for the Transformer.
    board values: 1 = current player, -1 = opponent, 0 = empty
    """
    encoded = np.zeros((BOARD_ROWS, BOARD_COLS, 2), dtype=np.float32)
    encoded[:, :, 0] = (board == 1)
    encoded[:, :, 1] = (board == -1)
    return extract_patches_transformer(encoded)


# ======================
# CNN MODEL
# ======================
class Connect4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, NUM_ACTIONS)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        x = torch.clamp(x, -10, 10)
        return F.softmax(x, dim=1)


# ======================
# TRANSFORMER MODEL
# ======================
class Connect4Transformer(nn.Module):
    def __init__(self, num_patches=42, patch_size=2, hidden_dim=128, num_layers=6,
                 num_heads=8, mlp_dim=256, dropout_rate=0.3, num_classes=7):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

        self.patch_embedding = nn.Linear(patch_size, hidden_dim)
        self.row_embedding = nn.Embedding(6, hidden_dim // 2)
        self.col_embedding = nn.Embedding(7, hidden_dim // 2)
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)

        row_indices = torch.arange(6, device=x.device).repeat_interleave(7)
        col_indices = torch.arange(7, device=x.device).repeat(6)
        row_emb = self.row_embedding(row_indices)
        col_emb = self.col_embedding(col_indices)
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        x = x + pos_emb.unsqueeze(0)

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.norm(x)
        x = self.head(x)
        return x


# ======================
# LOADING HELPERS
# ======================
def load_cnn(path="connect4_cnn__1_.pt"):
    """Load the CNN model from a state dict .pt file"""
    model = Connect4CNN().to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"CNN loaded from {path}")
    return model

def load_transformer(path="connect4_transformer__1_.pt"):
    """Load the Transformer model from a state dict .pt file"""
    model = Connect4Transformer().to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
    # If it was saved as full model, extract state dict
    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Transformer loaded from {path}")
    return model


# ======================
# PREDICT HELPERS
# ======================
def predict_move_cnn(model, board, legal_moves=None):
    """
    Returns move probabilities from CNN.
    board: numpy array (6x7), 1=current player, -1=opponent
    legal_moves: list of valid column indices (optional, masks illegal moves)
    """
    inp = encode_board_cnn(board)
    model.eval()
    with torch.no_grad():
        probs = model(inp)[0].cpu().numpy()  # shape (7,)
    if legal_moves is not None:
        mask = np.zeros(NUM_ACTIONS)
        mask[legal_moves] = 1
        probs = probs * mask
        probs = probs / probs.sum()
    return probs  # sample from this for PG, argmax for greedy

def predict_move_transformer(model, board, legal_moves=None):
    """
    Returns move probabilities from Transformer.
    board: numpy array (6x7), 1=current player, -1=opponent
    legal_moves: list of valid column indices (optional, masks illegal moves)
    """
    inp = encode_board_transformer(board)
    model.eval()
    with torch.no_grad():
        logits = model(inp)[0].cpu().numpy()  # shape (7,)
    probs = np.exp(logits) / np.exp(logits).sum()  # softmax
    if legal_moves is not None:
        mask = np.zeros(NUM_ACTIONS)
        mask[legal_moves] = 1
        probs = probs * mask
        probs = probs / probs.sum()
    return probs  # sample from this for PG, argmax for greedy
