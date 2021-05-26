import os
from pathlib import Path

class ImageClassificationConfig:
    '''
    ## Experiment Params: These are the experimental params which are
    usually constant for a Project
    '''
    Image_Depth = 3
    Project_name = 'nehtra_rot'
    Unstructured_Data_Path = ''
    Base_Data_Path = ''
    Path_Curr_Dir = Path(__file__).parent.absolute()
    Path_Parent_Dir = str(Path(Path_Curr_Dir).parents[0])
    Gt_Path = os.path.join(Path_Parent_Dir, 'input', 'cassava-leaf-disease-classification', 'train.csv')
    CheckPoints_Dir = os.path.join(Path_Parent_Dir, 'checkpoints')
    Img_Ext = '.jpg'
    Num_Classes = 360
    CheckPoints_Path = ''
    Mode = 'train'
    Validation_Fraction = 0.2
    Dataset_Shuffle = True
    Cyclic_LR = False
    No_System_Threads = 8
    Device = "cuda"
    Number_GPU = 1
    '''
     options are ResNet50, MobileNetV2
    '''
    Network_Architecture = 'ResNet50'
    '''
    ## Hyper Parameters: These are the standard tuning params 
    for an experiment.
    '''
    learning_rate = 0.0001
    batch_size_per_gpu = 16
    batch_size = batch_size_per_gpu * Number_GPU
    optimizer = 'adam'
    epochs = 200
    img_dim = 224
    img_channels = 3
    '''
    ## Callbacks and model-checkpoints parameters
    '''
    early_stopping_patience = 3
    model_chkpoint_period = 4
    one_cycle_lr_policy = True

ConfigObj = ImageClassificationConfig()
