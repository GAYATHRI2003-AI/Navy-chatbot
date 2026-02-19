# ML/DL Optimization Configuration
# ========================================

# CLIP Model Configuration
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_DEVICE = "cpu"  # Options: "cpu", "cuda:0", "auto"
CLIP_BATCH_SIZE = 8  # Batch process frames
CLIP_ENABLE_CACHING = True  # Enable embedding cache

# Acoustic Processing
ACOUSTIC_SR = 16000  # Sample rate
ACOUSTIC_N_MFCC = 13  # MFCC features
ACOUSTIC_N_MEL = 128  # Mel spectrogram features
ACOUSTIC_USE_REAL_FEATURES = True  # Extract real acoustic features

# LLM Configuration
LLM_ADAPTIVE_TIMEOUT = True  # Use adaptive timeouts
LLM_BASE_TIMEOUT = 25  # Base timeout in seconds
LLM_MAX_TIMEOUT = 60  # Maximum timeout
LLM_ENABLE_CACHE = True  # Cache LLM responses
LLM_CACHE_SIZE = 1000  # Max cache entries

# Vector Database
FAISS_AUTO_REBUILD = True  # Auto-rebuild on load failure
FAISS_MAX_RETRIES = 3  # Max rebuild attempts

# Model Quantization
ENABLE_QUANTIZATION = False  # Set to True for int8 quantization
QUANTIZATION_TYPE = "int8"  # Options: "int8", "float16"

# Context Pruning
MAX_CONTEXT_CHARS = 2800  # Max chars for LLM context
CONTEXT_RETENTION_RATIO = 0.85  # Keep 85% of context

# Knowledge Base Management
KB_AUTO_UPDATE = True  # Allow dynamic KB updates
KB_VALIDATION_ENABLED = True  # Validate facts before adding

# Profiling & Monitoring
ENABLE_PROFILING = True  # Profile performance metrics
PROFILE_LOG_FILE = "logs/ml_profile.json"
