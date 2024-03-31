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

def parse_args(config_path):
    # Default values for arguments
    args = argparse.Namespace(
        weights_dir='5b',
        model_save_dir='./logs',
        data_dir='./logs',
        freeze_encodec_decoder=True,
        clean_poses=True,
        movement_threshold=0.09,
        keypoints_threshold=3,
        frame_error_rate=0.08,
        batch_size=8,
        num_epochs=100,
        g_learning_rate=1e-5,
        d_learning_rate=1e-5,
        gradient_accumulation_steps=1,
        nll_loss_weight=1e-5,
        bce_loss_weight=1e-5,
        teacher_forcing_ratio=1e-5,
        val_epoch_interval=100,
        n_critic=100,
        encodec_model_id="facebook/encodec_24khz",
        sample_rate=24000,
        starting_weights_path=None,
        input_size=None,
        embed_size=32,
        train_ratio=0.8,
        val_ratio=0.2,
        random_seed=40,
        src_pad_idx=0,
        trg_pad_idx=0,
        num_layers=1,
        heads=2,
        dropout=0.1,
        pose2audio_num_layers=2,
        pose2audio_num_heads=2,
        pose2audio_dropout=0.1,
        clip_value=0.01,
        discriminator_hidden_units=32,
        discriminator_num_hidden_layers=2,
        discriminator_alpha=0.01,
        use_cuda=True,
        use_mps=False
    )

    # Dictionary to hold expected types for each argument
    expected_types = {arg: type(getattr(args, arg)) for arg in vars(args)}

    # Read and convert config file values
    if config_path:
        config_values = read_config(config_path, expected_types)
        for key, value in config_values.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args
