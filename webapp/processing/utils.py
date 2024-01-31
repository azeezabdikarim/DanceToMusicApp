import argparse

def read_config(config_file, expected_types):
    config = {}
    with open(config_file, 'r') as file:
        for line in file:
            if ("=" in line) and (not line.startswith("#")):
                key, value = line.strip().split(' = ')
                try:
                    if value == 'None':
                        config[key] = None
                    else:
                        config[key] = expected_types[key](value)
                except ValueError:
                    raise ValueError(f"Invalid type for {key}: expected {expected_types[key]}, got {type(value)}")    
    return config

def parse_args(config):
    parser = argparse.ArgumentParser()

    # Argument to specify the path to the configuration file
    parser.add_argument("--config", type=str, help="Path to the config file")

    # Other command line arguments with default values
    parser.add_argument("--weights_dir", default='5b')
    parser.add_argument("--model_save_dir", default='./logs')
    parser.add_argument("--data_dir", default='./logs')
    parser.add_argument("--freeze_encodec_decoder", type=bool, default=True, help="Flag to use freeze the encodec decoder. Set to false if you want to train the decoder")


    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--g_learning_rate", type=float, default=1e-5)
    parser.add_argument("--d_learning_rate", type=float, default=1e-5)

    parser.add_argument("--nll_loss_weight", type=float, default=1e-5)
    parser.add_argument("--bce_loss_weight", type=float, default=1e-5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1e-5)
    parser.add_argument("--val_epoch_interval", type=int, default=100)
    parser.add_argument("--n_critic", type=int, default=100)

    parser.add_argument("--encodec_model_id", type=str, default="facebook/encodec_24khz", help="ID of the encodec model.")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Sample rate for audio processing.")
    parser.add_argument("--starting_weights_path", type=str, default=None, help="Path to the starting weights file if doing trasnfer learning or restarting from a checkpoint.")
    parser.add_argument("--input_size", type=int, default=None, help="Input size for the model.")
    parser.add_argument("--embed_size", type=int, default=32, help="Embedding size for the model.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation data ratio.")
    parser.add_argument("--random_seed", type=int, default=40, help="Random seed for reproducibility.")
    parser.add_argument("--src_pad_idx", type=int, default=0, help="Source padding index.")
    parser.add_argument("--trg_pad_idx", type=int, default=0, help="Target padding index.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the pose2audio transfromer model.")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads in the pose2audio transfromer model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in the pose2audio transfromer model.")

    # Pose2AudioTransformer parameters
    parser.add_argument("--pose2audio_num_layers", type=int, default=2, help="Number of layers in the Pose2AudioTransformer model.")
    parser.add_argument("--pose2audio_num_heads", type=int, default=2, help="Number of attention heads in the Pose2AudioTransformer model.")
    parser.add_argument("--pose2audio_dropout", type=float, default=0.1, help="Dropout rate in the Pose2AudioTransformer model.")

    # Discriminator parameters
    parser.add_argument("--clip_value", type=float, default=0.01, help="Clipping value for weights in the discriminator.")
    parser.add_argument("--discriminator_hidden_units", type=int, default=32, help="Number of hidden units in the discriminator.")
    parser.add_argument("--discriminator_num_hidden_layers", type=int, default=2, help="Number of hidden layers in the discriminator.")
    parser.add_argument("--discriminator_alpha", type=float, default=0.01, help="Alpha parameter for the discriminator.")

    # Device configuration
    parser.add_argument("--use_cuda", type=bool, default=True, help="Flag to use CUDA if available.")
    parser.add_argument("--use_mps", type=bool, default=False, help="Flag to use Metal Performance Shaders (MPS) if available on macOS.")

    args = parser.parse_args()
    args.config = config

     # Dictionary to hold expected types for each argument
    expected_types = {}
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            expected_types[arg] = type(value)

    # Read and convert config file values
    if args.config:
        config_values = read_config(args.config, expected_types)
        for key, value in config_values.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args    
