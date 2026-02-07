"""Tests for audio conversion utilities."""

import numpy as np
import pytest

from agent import (
    DEEPGRAM_SAMPLE_RATE,
    ELEVENLABS_SAMPLE_RATE,
    PLIVO_SAMPLE_RATE,
    pcm_to_ulaw,
    resample_audio,
    ulaw_to_pcm,
)


class TestUlawConversion:
    """Tests for u-law encoding/decoding."""

    def test_ulaw_to_pcm_basic(self):
        """Test basic u-law to PCM conversion."""
        # u-law silence is 0xFF (127 after bias removal)
        ulaw_silence = bytes([0xFF] * 100)
        pcm = ulaw_to_pcm(ulaw_silence)

        # Should produce 16-bit PCM (2 bytes per sample)
        assert len(pcm) == 200

        # Silence should produce near-zero values
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert np.abs(samples).max() < 100

    def test_pcm_to_ulaw_basic(self):
        """Test basic PCM to u-law conversion."""
        # Create silent PCM (zeros)
        pcm_silence = np.zeros(100, dtype=np.int16).tobytes()
        ulaw = pcm_to_ulaw(pcm_silence)

        # Should produce same number of samples (1 byte each)
        assert len(ulaw) == 100

    def test_ulaw_pcm_roundtrip(self):
        """Test that u-law -> PCM -> u-law is reasonably accurate."""
        # Create some u-law data (avoiding edge cases at sign boundaries)
        # u-law values 0-127 are negative, 128-255 are positive
        original_ulaw = bytes([i for i in range(256) if i not in (127, 255)])

        # Convert to PCM and back
        pcm = ulaw_to_pcm(original_ulaw)
        recovered_ulaw = pcm_to_ulaw(pcm)

        # Most values should round-trip correctly
        # Allow some tolerance due to the encoding algorithm
        matches = sum(1 for a, b in zip(original_ulaw, recovered_ulaw) if a == b)
        match_ratio = matches / len(original_ulaw)

        # At least 90% should match exactly
        assert match_ratio > 0.9

    def test_pcm_ulaw_roundtrip(self):
        """Test that PCM -> u-law -> PCM preserves signal approximately."""
        # Create a sine wave
        t = np.linspace(0, 1, 8000)
        sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        original_pcm = sine.tobytes()

        # Convert to u-law and back
        ulaw = pcm_to_ulaw(original_pcm)
        recovered_pcm = ulaw_to_pcm(ulaw)

        # Compare signals (u-law is lossy, so allow some error)
        original = np.frombuffer(original_pcm, dtype=np.int16).astype(float)
        recovered = np.frombuffer(recovered_pcm, dtype=np.int16).astype(float)

        # Correlation should be very high
        correlation = np.corrcoef(original, recovered)[0, 1]
        assert correlation > 0.99

    def test_ulaw_to_pcm_empty(self):
        """Test u-law to PCM with empty input."""
        result = ulaw_to_pcm(b"")
        assert result == b""

    def test_pcm_to_ulaw_empty(self):
        """Test PCM to u-law with empty input."""
        result = pcm_to_ulaw(b"")
        assert result == b""

    def test_ulaw_values_in_range(self):
        """Test that u-law output values are valid bytes."""
        # Create PCM with full dynamic range
        pcm = np.array([-32768, -16384, 0, 16383, 32767], dtype=np.int16).tobytes()
        ulaw = pcm_to_ulaw(pcm)

        # All values should be valid bytes
        assert all(0 <= b <= 255 for b in ulaw)


class TestResampling:
    """Tests for audio resampling."""

    def test_resample_same_rate(self):
        """Test that same-rate resampling returns input unchanged."""
        original = np.random.randint(-32768, 32767, 1000, dtype=np.int16).tobytes()
        result = resample_audio(original, 8000, 8000)
        assert result == original

    def test_resample_upsample(self):
        """Test upsampling from 8kHz to 24kHz."""
        # 80 samples at 8kHz = 10ms
        original = np.random.randint(-32768, 32767, 80, dtype=np.int16).tobytes()
        result = resample_audio(original, 8000, 24000)

        # Should have 3x the samples
        result_samples = np.frombuffer(result, dtype=np.int16)
        assert len(result_samples) == 240

    def test_resample_downsample(self):
        """Test downsampling from 24kHz to 8kHz."""
        # 240 samples at 24kHz = 10ms
        original = np.random.randint(-32768, 32767, 240, dtype=np.int16).tobytes()
        result = resample_audio(original, 24000, 8000)

        # Should have 1/3 the samples
        result_samples = np.frombuffer(result, dtype=np.int16)
        assert len(result_samples) == 80

    def test_resample_preserves_frequency(self):
        """Test that resampling preserves signal frequency."""
        # Create 440Hz sine at 8kHz
        duration = 0.1  # 100ms
        samples_8k = int(8000 * duration)
        t_8k = np.linspace(0, duration, samples_8k, endpoint=False)
        sine_8k = (np.sin(2 * np.pi * 440 * t_8k) * 16000).astype(np.int16)

        # Upsample to 24kHz
        resampled = resample_audio(sine_8k.tobytes(), 8000, 24000)
        samples_24k = np.frombuffer(resampled, dtype=np.int16)

        # Should have 3x samples
        assert len(samples_24k) == samples_8k * 3

        # Check frequency is preserved via FFT
        fft = np.abs(np.fft.fft(samples_24k.astype(float)))
        freqs = np.fft.fftfreq(len(samples_24k), 1/24000)
        peak_freq = abs(freqs[np.argmax(fft[:len(fft)//2])])

        # Peak should be around 440Hz (within 10Hz tolerance)
        assert abs(peak_freq - 440) < 10

    def test_resample_empty(self):
        """Test resampling with empty input."""
        result = resample_audio(b"", 8000, 24000)
        assert result == b""

    def test_resample_values_in_range(self):
        """Test that resampled values stay within int16 range."""
        # Use values near the limits
        original = np.array([-32768, 32767] * 50, dtype=np.int16).tobytes()
        result = resample_audio(original, 8000, 24000)

        samples = np.frombuffer(result, dtype=np.int16)
        assert samples.min() >= -32768
        assert samples.max() <= 32767


class TestAudioPipeline:
    """Tests for the full audio conversion pipeline."""

    def test_plivo_to_deepgram_format(self):
        """Test Plivo (u-law 8kHz) to Deepgram (PCM 8kHz) conversion."""
        # Simulate Plivo audio
        ulaw_audio = bytes([0x80 + i % 128 for i in range(160)])  # 20ms at 8kHz

        # Convert to Deepgram format (just PCM decode, same sample rate)
        pcm_audio = ulaw_to_pcm(ulaw_audio)

        # Deepgram expects linear16 at 8kHz
        assert len(pcm_audio) == 320  # 160 samples * 2 bytes

    def test_elevenlabs_to_plivo_format(self):
        """Test ElevenLabs (PCM 24kHz) to Plivo (u-law 8kHz) conversion."""
        # Simulate ElevenLabs TTS output (24kHz PCM)
        duration_ms = 20
        samples_24k = int(ELEVENLABS_SAMPLE_RATE * duration_ms / 1000)
        elevenlabs_audio = np.random.randint(
            -16000, 16000, samples_24k, dtype=np.int16
        ).tobytes()

        # Downsample to 8kHz
        pcm_8k = resample_audio(elevenlabs_audio, ELEVENLABS_SAMPLE_RATE, PLIVO_SAMPLE_RATE)

        # Convert to u-law
        ulaw_audio = pcm_to_ulaw(pcm_8k)

        # Should have correct number of samples for Plivo
        expected_samples = int(PLIVO_SAMPLE_RATE * duration_ms / 1000)
        assert len(ulaw_audio) == expected_samples

    def test_full_roundtrip(self):
        """Test full audio pipeline roundtrip."""
        # Create test audio at Plivo format
        original_ulaw = bytes([0x80 + (i * 7) % 128 for i in range(800)])  # 100ms

        # Plivo -> Deepgram (PCM 8kHz)
        pcm_8k = ulaw_to_pcm(original_ulaw)

        # Simulate ElevenLabs output at 24kHz
        pcm_24k = resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, ELEVENLABS_SAMPLE_RATE)

        # ElevenLabs -> Plivo
        pcm_8k_out = resample_audio(pcm_24k, ELEVENLABS_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
        final_ulaw = pcm_to_ulaw(pcm_8k_out)

        # Should have same length
        assert len(final_ulaw) == len(original_ulaw)

        # Convert both to PCM for comparison
        original_pcm = np.frombuffer(ulaw_to_pcm(original_ulaw), dtype=np.int16).astype(float)
        final_pcm = np.frombuffer(ulaw_to_pcm(final_ulaw), dtype=np.int16).astype(float)

        # Correlation should be high (allowing for resampling artifacts)
        correlation = np.corrcoef(original_pcm, final_pcm)[0, 1]
        assert correlation > 0.95
