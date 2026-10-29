class ModelConfig:
    # model config
    NUM_VEC = 512
    DIM_VEC = 384
    DIM_LATENT = 32
    RESOLUTION = 512
    DIM_FEATURE_FIELD = 64

    # pos emb config
    POS_EMB_FREQS = 6
    POS_EMB_INCLUDE_INPUT = False
    POS_EMB_LAYERS = 3

    # encoder config
    ENC_SA_BLOCKS = 6  
    ENC_SA_EMB = 256       
    ENC_SA_HEADS = 8 
    ENC_SA_EXPANSION = 4      
    ENC_CA_BLOCKS = 1
    ENC_CA_EMB = 128    
    ENC_CA_HEADS = 4   
    ENC_CA_EXPANSION = 2

    # decoder config
    DEC_SA_BLOCKS = 3  
    DEC_SA_EMB = 256       
    DEC_SA_HEADS = 8 
    DEC_SA_EXPANSION = 4      
    DEC_CA_BLOCKS = 2
    DEC_CA_EMB = 128    
    DEC_CA_HEADS = 4   
    DEC_CA_EXPANSION = 2


class TrainConfig:
    # dataset config
    SAMPLE_FCT = 4096
    SAMPLE_NEG = 2048
    HARD_START_EPOCH = 1 
    HARD_END_EPOCH = 32 
    HARD_PER_EPOCH = 32
    HARD_TARGET = (HARD_END_EPOCH - HARD_START_EPOCH + 1) * HARD_PER_EPOCH
    SHELL_RADIUS = 30 
    TRAIN_RATIO = 0.9
    VAL_RATIO = 1.0 - TRAIN_RATIO
    RANDOM_SEED = 42

    # dataloader config
    TRAIN_BATCH_SIZE = 4
    TRAIN_SHUFFLE = True
    TRAIN_NUM_WORKERS = 4
    TRAIN_PIN_MEMORY = True
    TRAIN_DROP_LAST = True          

    VAL_BATCH_SIZE = 8 
    VAL_SHUFFLE = False           
    VAL_NUM_WORKERS = 4            
    VAL_PIN_MEMORY = True
    VAL_DROP_LAST = False 

    # loop config
    EPOCHS = 100
    LOG_INTERVAL = 100        
    VAL_INTERVAL = 5         
    SAVE_INTERVAL = 10         

    # optimizer config
    GRAD_CLIP = 1.0 #0.5
    LR = 5e-5                     
    WEIGHT_DECAY = 1e-5

    WARMUP_START_FACTOR = 0.1       
    WARMUP_END_FACTOR = 1.0         
    WARMUP_ITERS = 10 

    COSINE_T_MAX = EPOCHS - WARMUP_ITERS      
    SCHEDULER_MILESTONES = [WARMUP_ITERS]        
    COSINE_ETA_MIN = 1e-6 

    # loss config
    BALANCE_ACTIVE = 2.85
    BALANCE_ANCHOR = 1.00
    BALANCE_NORMAL = 1.61
    BALANCE_FLUX = 0.82

    PRIORITY_ACTIVE = 1.0
    PRIORITY_ANCHOR = 1.0
    PRIORITY_NORMAL = 1.0 #1.2
    PRIORITY_FLUX = 1.0 #3.25 

    LAMBDA_ACTIVE = BALANCE_ACTIVE * PRIORITY_ACTIVE
    LAMBDA_ANCHOR = BALANCE_ANCHOR * PRIORITY_ANCHOR
    LAMBDA_NORMAL = BALANCE_NORMAL * PRIORITY_NORMAL
    LAMBDA_FLUX = BALANCE_FLUX * PRIORITY_FLUX

    KL_ANNEAL_EPOCHS = 40
    KL_TARGET_WEIGHT = 0.1


class TestConfig:
    # dataloader config
    TEST_BATCH_SIZE = 4
    TEST_SHUFFLE = False
    TEST_NUM_WORKERS = 4
    TEST_PIN_MEMORY = True
    TEST_DROP_LAST = False
