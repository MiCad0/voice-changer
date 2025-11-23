import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
import warnings

# Suppress resource warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SAMPLE_RATE = 48000
BLOCK_SIZE = 16384   # Paquet plus gros = plus de temps pour l'IA, mais plus de latence
CHANNELS_IN = 1     # Micro (Mono)
CHANNELS_OUT = 2    # Casque/Câble (Stéréo)
FADE_SIZE = 256     # Zone de lissage (Anti-click). 256 samples @ 48kHz = ~5ms de fondu

# --- FILES D'ATTENTE (BUFFERS) ---
input_queue = queue.Queue()  # Les paquets bruts venant du micro attendent ici
output_queue = queue.Queue() # Les paquets transformés attendent ici d'être joués

# Choix du processeur
print("Is CUDA available:", torch.cuda.is_available())
DEVICE = torch.device("cpu")
print(f"--- Moteur de traitement : {DEVICE} ---")


# --- Paramètres du Voice Changer ---
PITCH_SHIFT_STEPS = 0  # Ajustez cette valeur pour changer la voix !

@torch.inference_mode()
def ma_fonction_de_transformation_ia(audio_tensor):
    """
    Vrai Pitch Shifter (Phase Vocoder) accéléré par GPU.
    """
    # 1. Transfert vers le GPU
    wave_gpu = audio_tensor.to(DEVICE).unsqueeze(0) 

    # 2. Paramètres DSP
    # IMPORTANT : n_fft doit être <= BLOCK_SIZE
    n_fft = BLOCK_SIZE 
    hop_length = 512
    window = torch.hann_window(n_fft).to(DEVICE)
    
    # 3. STFT
    stft_matrix = torch.stft(
        wave_gpu, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=window, 
        return_complex=True,
        center=True 
    )

    # 4. Pitch Shifting
    magnitude = torch.abs(stft_matrix)
    phase = torch.angle(stft_matrix)
    
    scale_factor = 2 ** (PITCH_SHIFT_STEPS / 12.0)
    n_bins = magnitude.shape[1]
    
    indices = torch.arange(n_bins, device=DEVICE)
    source_indices = (indices / scale_factor).long()
    
    # Masque de sécurité
    mask = (source_indices < n_bins)
    
    new_magnitude = torch.zeros_like(magnitude)
    new_phase = torch.zeros_like(phase)
    
    # Interpolation Nearest Neighbor
    new_magnitude[:, indices[mask], :] = magnitude[:, source_indices[mask], :]
    new_phase[:, indices[mask], :] = phase[:, source_indices[mask], :]
    
    new_stft = torch.polar(new_magnitude, new_phase)

    # 5. ISTFT
    output_wave = torch.istft(
        new_stft, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=window, 
        center=True,
        length=wave_gpu.shape[1]
    )

    return output_wave.squeeze(0).cpu()

def callback_audio(indata, outdata, frames, time_info, status):
    """
    Callback Audio temps réel (Thread Haute Priorité)
    """
    if status:
        print(f"Status Audio: {status}", file=sys.stderr)

    # 1. ENREGISTREMENT
    input_queue.put(indata.copy())

    # 2. LECTURE
    try:
        data_processed = output_queue.get_nowait()
        
        if data_processed.shape == outdata.shape:
            outdata[:] = data_processed
        else:
            # Gestion Mono -> Stéréo
            if outdata.shape[1] == 2 and data_processed.ndim == 1:
                outdata[:] = data_processed[:, np.newaxis]
            else:
                outdata[:] = np.zeros_like(outdata)

    except queue.Empty:
        # Silence si l'IA n'est pas prête (Underrun)
        outdata[:] = 0

def boucle_principale_traitement():
    """
    Thread de Traitement IA (Avec Cross-Fading Anti-Click)
    """
    print("[Processing] Boucle de traitement IA démarrée.")
    
    # --- Initialisation du Cross-Fade ---
    previous_chunk_end = None
    
    # On pré-calcule les courbes de fondu pour ne pas perdre de temps CPU
    fade_in_curve = np.linspace(0, 1, FADE_SIZE)
    fade_out_curve = 1.0 - fade_in_curve
    
    while True:
        # 1. Récupération (Bloquant)
        raw_data_numpy = input_queue.get()
        
        # 2. Traitement IA
        tensor_in = torch.from_numpy(raw_data_numpy[:, 0]).float()
        
        try:
            tensor_out = ma_fonction_de_transformation_ia(tensor_in)
            processed_numpy = tensor_out.numpy()
            
            # --- 3. LOGIQUE DE CROSS-FADING (ANTI-CLICK) ---
            # Si nous avons un historique du paquet précédent
            if previous_chunk_end is not None:
                # On prend le début du nouveau paquet
                start_of_new = processed_numpy[:FADE_SIZE]
                # On prend la fin de l'ancien paquet
                end_of_old = previous_chunk_end
                
                # On mélange les deux (Moyenne pondérée)
                # Ancien son s'éteint + Nouveau son s'allume
                smoothed_transition = (start_of_new * fade_in_curve) + (end_of_old * fade_out_curve)
                
                # On applique la transition au nouveau paquet
                processed_numpy[:FADE_SIZE] = smoothed_transition
            
            # On sauvegarde la FIN de ce paquet actuel pour le prochain tour
            previous_chunk_end = processed_numpy[-FADE_SIZE:].copy()
            # -----------------------------------------------
            
            # 4. Préparation sortie Stéréo
            final_output = np.zeros((len(processed_numpy), CHANNELS_OUT), dtype='float32')
            final_output[:, 0] = processed_numpy
            final_output[:, 1] = processed_numpy
            
            # 5. Envoi
            output_queue.put(final_output)
            
        except Exception as e:
            print(f"Erreur de traitement : {e}")

if __name__ == "__main__":
    print("--- Voice Changer Multithreadé (Avec Fade Anti-Click) ---")
    
    # Tentative d'augmentation de priorité (Windows)
    try:
        import os, psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.REALTIME_PRIORITY_CLASS)
    except:
        pass

    print(sd.query_devices())
    try:
        in_id = int(input("\nID Micro (Input) : "))
        out_id = int(input("ID Sortie (CABLE Input) : "))
    except:
        sys.exit(0)

    # Lancement du Thread IA
    processing_thread = threading.Thread(target=boucle_principale_traitement, daemon=True)
    processing_thread.start()

    # Lancement du Flux Audio
    print(f"\n[Audio] Démarrage du flux sur ID {in_id} -> {out_id}")
    print("Parlez dans le micro. Appuyez sur Ctrl+C pour arrêter.")
    
    try:
        with sd.Stream(device=(in_id, out_id),
                       samplerate=SAMPLE_RATE,
                       blocksize=BLOCK_SIZE,
                       channels=(CHANNELS_IN, CHANNELS_OUT),
                       dtype='float32',
                       callback=callback_audio):
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nArrêt...")
    except Exception as e:
        print(f"Erreur : {e}")