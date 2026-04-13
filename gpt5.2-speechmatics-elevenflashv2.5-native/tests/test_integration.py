"""
Integration tests for GPT 5.2 Mini + Speechmatics + ElevenLabs Voice Agent.

Test Levels:
1. Unit Tests - Test individual components (audio conversion, phone normalization)
2. Local Integration - Test WebSocket flow without external services
3. API Integration - Test OpenAI, Speechmatics, ElevenLabs API connections
4. Plivo Integration - Test Plivo API configuration

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
"""

from __future__ import annotations

import math
import struct

import numpy as np
import phonenumbers

from utils import (
    normalize_phone_number,
    pcm_to_ulaw,
    plivo_to_speechmatics,
    plivo_to_vad,
    ulaw_to_pcm,
)

# =============================================================================
# UNIT TESTS - Test individual components
# =============================================================================


class TestUnitAudioConversion:
    """Unit tests for audio format conversion."""

    def test_ulaw_to_pcm_conversion(self):
        """Test μ-law to PCM conversion produces expected output size and low amplitude silence."""
        ulaw_silence = b"\xff" * 160
        pcm_audio = ulaw_to_pcm(ulaw_silence)

        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)
        avg_amplitude = sum(abs(s) for s in samples) / len(samples)

        assert len(pcm_audio) == 320  # 160 samples * 2 bytes
        assert avg_amplitude < 100  # Should be near silence

    def test_pcm_to_ulaw_conversion(self):
        """Test PCM to μ-law conversion produces half-size output."""
        pcm_silence = b"\x00" * 320
        ulaw_audio = pcm_to_ulaw(pcm_silence)

        assert len(ulaw_audio) == 160  # Half the size

    def test_audio_roundtrip(self):
        """Test that audio survives roundtrip conversion with correlation > 0.9."""
        samples = []
        for i in range(160):
            sample = int(16000 * math.sin(2 * math.pi * 440 * i / 8000))
            samples.append(sample)
        pcm_original = struct.pack(f"{len(samples)}h", *samples)

        ulaw = pcm_to_ulaw(pcm_original)
        pcm_restored = ulaw_to_pcm(ulaw)

        original_samples = struct.unpack(f"{len(pcm_original) // 2}h", pcm_original)
        restored_samples = struct.unpack(f"{len(pcm_restored) // 2}h", pcm_restored)

        # Check correlation (should be > 0.9)
        correlation = sum(o * r for o, r in zip(original_samples, restored_samples, strict=True))
        orig_energy = sum(o * o for o in original_samples)
        rest_energy = sum(r * r for r in restored_samples)

        if orig_energy > 0 and rest_energy > 0:
            normalized_corr = correlation / (orig_energy * rest_energy) ** 0.5
            assert normalized_corr > 0.9, "Audio quality degraded too much"

    def test_plivo_to_speechmatics_no_resample(self):
        """plivo_to_speechmatics converts μ-law 8kHz to PCM16 8kHz (no resample)."""
        # 160 bytes μ-law = 20ms at 8kHz (one Plivo packet)
        mulaw_data = b"\xff" * 160
        result = plivo_to_speechmatics(mulaw_data)

        # Should be PCM16 at 8kHz — same number of samples, 2 bytes each
        assert len(result) == 320  # 160 samples * 2 bytes
        # Should match raw ulaw_to_pcm (no resampling)
        assert result == ulaw_to_pcm(mulaw_data)

    def test_plivo_to_vad_resamples_to_16k(self):
        """plivo_to_vad returns float32 numpy array at 16kHz (2x input samples)."""
        mulaw_data = b"\xff" * 160
        result = plivo_to_vad(mulaw_data)

        # Output should be a float32 numpy array
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        # 160 samples at 8kHz -> 320 samples at 16kHz (2x)
        assert len(result) == 320


# =============================================================================
# UNIT TESTS - Phone number normalization
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        """Test normalizing E.164 formatted numbers."""
        phone = "+16572338892"
        parsed = phonenumbers.parse(phone, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

        assert e164 == "+16572338892"

    def test_normalize_with_spaces(self):
        """Test normalizing numbers with spaces."""
        phone = "+1 657-233-8892"
        parsed = phonenumbers.parse(phone, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

        assert e164 == "+16572338892"

    def test_normalize_local_format(self):
        """Test normalizing local format numbers."""
        result = normalize_phone_number("(657) 233-8892", "US")
        assert result == "16572338892"
