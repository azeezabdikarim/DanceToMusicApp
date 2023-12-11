import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import time

import librosa
import torch
import torchaudio
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.optim import Adam
from transformers import EncodecModel
from models import Pose2AudioTransformer, AudioDescriminator
from utils import DanceToMusic
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split


# Define a function to save the model and delete the last saved model
def save_model(model, folder_path, loss, last_saved_model, name = ''):
    model_name = f"{name}_best_model_{loss:.4f}.pt"
    model_path = os.path.join(folder_path, model_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(model.state_dict(), model_path)
    if last_saved_model:
        try:
            os.remove(os.path.join(folder_path, last_saved_model))
        except FileNotFoundError:
            print(f"Warning: Could not find {last_saved_model} to delete.")
    return model_name

def main():
    # assign GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")


    model_id = "facebook/encodec_24khz"
    encodec_model = EncodecModel.from_pretrained(model_id)
    codebook_size = encodec_model.quantizer.codebook_size
    encodec_model.to(device)
    # processor = AutoProcessor.from_pretrained(model_id)
    
    sample_rate = 24000
    batch_size = 1 # Batch size for Nvididias GTX 3080 9.88/10GB

    # data_dir = '/home/azeez/Documents/projects/DanceToMusicApp/ml/data/samples/5sec_expando_dnb_min_training_data'
    data_dir = "/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_dnb_min_training_data"
    data_dir = '/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_test'
    # data_dir = '/home/azeez/Documents/projects/DanceToMusicApp/ml/data/samples/5sec_expando_dnb_min_training_data'
    train_dataset = DanceToMusic(data_dir, encoder = encodec_model, sample_rate = sample_rate, device=device, dnb = True)
    embed_size = train_dataset.data['poses'].shape[2] * train_dataset.data['poses'].shape[3]

    target_shape = train_dataset.data['audio_codes'][0].shape
    descriminator = AudioDescriminator(target_shape, hidden_units = 64, num_hidden_layers = 3, alpha = 0.01)
    descriminator.to(device)


    train_ratio = 0.8 # Define the split ratio
    val_ratio = 1 - train_ratio
    dataset_size = len(train_dataset)
    train_len = int(train_ratio * dataset_size)
    val_len = dataset_size - train_len

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    src_pad_idx = 0
    trg_pad_idx = 0
    # device = torch.device("cpu")
    pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, device=device, num_layers=4, heads = 4, embed_size=embed_size, dropout=0.1)
    pose_model.to(device)
    
    # weights = 'DanceToMusicApp/ml/model_weights/5_sec_dnb_best_model_weights_loss_4.911053791451962.pth'
    # weights = '/home/azeez/Documents/projects/DanceToMusicApp/ml/model_weights/5_sec_dnb_best_model_weights_loss_4.911053791451962.pth'
    # pose_model.load_state_dict(torch.load(weights, map_location=device))

    learning_rate = 1e-3
    criterion_g = torch.nn.NLLLoss()
    criterion_d = torch.nn.BCELoss()

    # criterion = MSELoss()
    optimizer_g = torch.optim.Adam(pose_model.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(descriminator.parameters(), lr=learning_rate)


    # Set up for tracking the best model
    # weights_dir = '/home/azeez/Documents/projects/DanceToMusicApp/ml/model_weights'
    weights_dir = '/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/model_weights'
    best_loss = float('inf')  # Initialize with a high value
    g_last_saved_model = ''
    d_last_saved_model = ''

    # Generate a unique directory name based on current date and time
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('./my_logs', 'run_' + current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    accumulation_steps = 12  # Number of mini-batches for gradient accumulation
    samples_processed = 0  # Keep track of samples processed so far within accumulation steps
    accumulated_loss = 0  # Initialize accumulated loss

    teacher_forcing_ratio = 0.5 # 50% of the time we will use teacher forcing
    val_epoch_interval = 5
    num_epochs = 30000
    for epoch in range(num_epochs):
        pose_model.train()
        descriminator.train()
        epoch_loss = 0  # Initialize epoch_loss
        timesteps = 0

        total_loss_g = 0  # Total generator loss
        total_loss_d = 0  # Total discriminator loss
        total_steps = 0  # To track the number of steps

        for i, (audio_codes, pose, pose_mask, wav, wav_mask, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer_g.zero_grad()  # Clear gradients
            
            # Forward pass
            wav = wav.unsqueeze(1)
            pose = pose
            pose_mask = pose_mask
            target = audio_codes.to(device)
            target_for_loss = target[:, 1:, :]
            
            input_for_next_step = target[:, 0:1, :]
            total_nll_loss = 0
            
            src = pose.to(device)
            enc_mask = pose_model.make_src_mask(src)
            trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))
            
            B, N, _, _ = pose.shape
            enc_context = pose_model.encoder(src.view(B, N, -1), enc_mask)
            
            for t in range(1, target.shape[1]):
                output_softmax, output_argmax, offset, _ = pose_model.decoder(input_for_next_step.to(device), enc_context, enc_mask, trg_mask)
                
                log_softmax_output = torch.log(output_softmax[:,-2:,:])
                log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])
                reshaped_target = target[:, t, :].reshape(-1).long()
                time_step_nll_loss = criterion_g(log_softmax_output_reshape, reshaped_target)
                total_nll_loss += time_step_nll_loss
                
                use_teacher_forcing = np.random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_for_next_step = torch.cat([input_for_next_step, target[:, t:t+1, :]], dim=1)  # Concatenate along the sequence length dimension
                else:
                    next_token = output_softmax[:,-2:,:].argmax(dim=2)
                    input_for_next_step = torch.cat([input_for_next_step, next_token.unsqueeze(1)], dim=1)
                trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))
                
                timesteps += 1

            fake_data = input_for_next_step
            # Train Discriminator on Real Data
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            real_output = descriminator(audio_codes)
            loss_d_real = criterion_d(real_output, real_labels)

            # Train Discriminator on Fake Data
            fake_labels = torch.zeros(batch_size, 1, device=device)
            fake_output = descriminator(fake_data.detach())
            loss_d_fake = criterion_d(fake_output, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # Update Generator based on Discriminator's feedback
            optimizer_g.zero_grad()
            output = descriminator(fake_data)
            loss_g = criterion_d(output, real_labels) # Train to fool the discriminator
            avg_nll_loss = total_nll_loss / (target.shape[1] - 1)   
            loss_g = loss_g + avg_nll_loss
            loss_g.backward()
            optimizer_g.step()

            total_loss_g += loss_g.item() + avg_nll_loss.item()
            total_loss_d += loss_d.item()
            total_steps += 1

            epoch_loss += (total_loss_g + total_loss_d)
            # Average losses for the epoch
            avg_batch_loss_g = total_loss_g / total_steps
            avg_batch_loss_d = total_loss_d / total_steps
            
            if total_steps % 5 == 0:
                writer.add_scalar('Generator Loss', avg_batch_loss_g, epoch)
                writer.add_scalar('Discriminator Loss', avg_batch_loss_d, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {avg_batch_loss_g:.4f}, Discriminator Loss: {avg_batch_loss_d:.4f}")

        avg_epoch_loss = epoch_loss / (timesteps)   # Compute average epoch loss
        # Save the best model, by checking if the new model is perfroming better than the previous best model
        if total_loss_g < best_loss:
            best_loss = total_loss_g
            g_last_saved_model = save_model(pose_model, weights_dir, best_loss, g_last_saved_model, name='gen_5_sec_dnb_')
            d_last_saved_model = save_model(descriminator, weights_dir, best_loss, d_last_saved_model, name='disc_5_sec_dnb_')

        if epoch % val_epoch_interval == 0:
            pose_model.eval()
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for audio_codes, pose, pose_mask, wav, wav_mask, _, _ in val_loader:
                    target = audio_codes.to(device)
                    input_for_next_step = target[:, 0:1, :]
                    
                    src = pose.to(device)
                    enc_mask = pose_model.make_src_mask(src)
                    trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))

                    B, N, _, _ = pose.shape
                    enc_context = pose_model.encoder(src.view(B, N, -1), enc_mask)
                    total_nll_loss = 0

                    for t in range(1, target.shape[1]):
                        output_softmax, output_argmax, offset, _ = pose_model.decoder(input_for_next_step.to(device), enc_context, enc_mask, trg_mask)

                        log_softmax_output = torch.log(output_softmax[:,-2:,:])
                        log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])
                        reshaped_target = target[:, t, :].reshape(-1).long()
                        time_step_nll_loss = criterion_g(log_softmax_output_reshape, reshaped_target)
                        total_nll_loss += time_step_nll_loss

                        next_token = output_softmax[:,-2:,:].argmax(dim=2)
                        input_for_next_step = torch.cat([input_for_next_step, next_token.unsqueeze(1)], dim=1)
                        trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))

                    avg_nll_loss = total_nll_loss / (target.shape[1] - 1)
                    val_loss += avg_nll_loss.item()
                    val_steps += 1

                avg_val_loss = val_loss / val_steps
                writer.add_scalar('Validation Loss', avg_val_loss, epoch)
                print(f"\n Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        else:
            print(f"\n Epoch [{epoch+1}/{num_epochs}]")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()


if __name__ == "__main__":
    main()