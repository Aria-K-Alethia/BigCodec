import torch
import torchaudio

def stft(x, fft_size, hop_size, win_length, window, use_complex=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(x, fft_size, hop_size, win_length, window.to(x.device),
                        return_complex=True)

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(torch.clamp(
            x_stft.real ** 2 + x_stft.imag ** 2, min=1e-7, max=1e3)).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3) # [B, 2, T, F]
        return res
