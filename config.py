class Config(object):
    # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    PRETRAINED = False
    NUM_CLASSES = 8

    # train config
    EPOCHS = 600
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 1e-3
    
    # image size
    IMAGE_SIZE = (128, 128)    #768x256,1024x384,1536x512
    
    # batch size  *len(devices)
    TRAIN_BATCH = 2
    VAL_BATCH = 2

    SAVE_INTERVAL = 50