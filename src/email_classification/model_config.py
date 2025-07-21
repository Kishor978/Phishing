# email_classification/config/model_configs.py

# --- Data and Preprocessing ---
DATA_PATH_TREC_05 = r'E:\Phising_detection\dataset\emails\TREC_05.csv'
DATA_PATH_TREC_06 = r'E:\Phising_detection\dataset\emails\TREC_06.csv'
DATA_PATH_TREC_07 = r'E:\Phising_detection\dataset\emails\TREC_07.csv'
PROCESSED_DATA_PATH = r'E:\Phising_detection\dataset\emails\processed\merged_emails.csv'

# Common Text Preprocessing
SPACY_MODEL = "en_core_web_sm" # SpaCy model to load
SPACY_MAX_LENGTH = 3_000_000 # Increased limit for large documents

# --- Training Parameters (Common for PyTorch models) ---
BATCH_SIZE = 64
EPOCHS = 20# Adjusted to 10 as seen common in notebooks, some had 20 but 10 is also used
LEARNING_RATE = 1e-3
PATIENCE = 3 # Common patience value for early stopping (e.g., in BiLSTM)

# --- BiLSTM Model Parameters ---
BILSTM_EMBED_DIM = 100
BILSTM_HIDDEN_DIM = 128
BILSTM_NUM_LAYERS = 2
BILSTM_DROPOUT = 0.5
BILSTM_BIDIRECTIONAL = True

# --- Dual Encoder BiLSTM Model Parameters ---
DUAL_BILSTM_EMBED_DIM = 100
DUAL_BILSTM_HIDDEN_DIM = 128
DUAL_BILSTM_NUM_LAYERS = 2
DUAL_BILSTM_DROPOUT = 0.5
DUAL_BILSTM_BIDIRECTIONAL = True
# Fusion layer parameters (derived from BiLSTM output * 2 for bidirectional, * 2 for dual encoders)
DUAL_BILSTM_FUSION_INPUT_DIM = DUAL_BILSTM_HIDDEN_DIM * 2 * 2 # 128 * 2 (bi) * 2 (sub+body) = 512
DUAL_BILSTM_FUSION_HIDDEN_DIM = 256 # Not explicitly specified, common choice
DUAL_BILSTM_FUSION_DROPOUT = 0.3 # Not explicitly specified, common choice

# --- Dual Encoder Transformer (Attention) Model Parameters ---
DUAL_TRANS_EMBED_DIM = 100
DUAL_TRANS_HIDDEN_DIM = 128 # Used for LSTM encoders before attention
DUAL_TRANS_NUM_LAYERS = 2
DUAL_TRANS_DROPOUT = 0.5
DUAL_TRANS_BIDIRECTIONAL = True
DUAL_TRANS_ATTN_HEADS = 1 # Not explicitly specified, common choice for single head
# Fusion layer parameters (derived from LSTM output * 2 for bidirectional, * 2 for dual encoders)
DUAL_TRANS_FUSION_INPUT_DIM = DUAL_TRANS_HIDDEN_DIM * 2 * 2 # 128 * 2 (bi) * 2 (sub+body) = 512
DUAL_TRANS_FUSION_HIDDEN_DIM = 256 # Not explicitly specified, common choice
DUAL_TRANS_FUSION_DROPOUT = 0.3 # Not explicitly specified, common choice

# --- Baseline Model Parameters (Random Forest) ---
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

# TF-IDF Vectorizer Parameters
TFIDF_MAX_FEATURES = 5000 # Arbitrary, tune if necessary, often not specified in notebooks
# (Note: notebooks used unspecified max_features or different ones for subject/body)
# Here, I'm using a general value. If specific max_features are needed per vectorizer,
# they should be defined in the traditional_models.py itself or here as dicts.