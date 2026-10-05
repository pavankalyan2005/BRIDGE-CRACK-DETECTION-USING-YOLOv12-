"""
Bridge Structural Health Monitoring (SHM) IoT Sensor Simulator
Simulates live sensor telemetry (Accelerometers, Strain Gauges, LVDT Deflection, Tilt)
and computes Fast Fourier Transform (FFT) Power Spectral Density for modal frequency shifts.
"""

import time
import math
import numpy as np
from typing import Dict, Any, List

class BridgeTelemetrySimulator:
    def __init__(self, sampling_rate_hz: int = 100, buffer_seconds: int = 2):
        self.sampling_rate = sampling_rate_hz
        self.num_samples = sampling_rate_hz * buffer_seconds
        self.time_array = np.linspace(0, buffer_seconds, self.num_samples)
        
        # Baseline healthy bridge modal frequencies (Hz)
        self.base_f1 = 2.45   # 1st Bending Mode
        self.base_f2 = 5.80   # 2nd Bending Mode
        self.base_f3 = 11.20  # Torsional Mode

    def generate_telemetry_frame(self, crack_severity_ratio: float = 0.0) -> Dict[str, Any]:
        """
        Generates simulated real-time telemetry correlated with visual crack severity (0.0 to 1.0).
        Stiffness loss causes modal frequency reduction: f = f0 * sqrt(1 - 0.25 * severity)
        """
        # Frequency shift due to structural stiffness degradation
        stiffness_factor = math.sqrt(max(0.65, 1.0 - 0.35 * crack_severity_ratio))
        f1 = self.base_f1 * stiffness_factor
        f2 = self.base_f2 * stiffness_factor
        f3 = self.base_f3 * stiffness_factor

        current_time = time.time()
        
        # Synthetic acceleration signal (g) with traffic pulses and noise
        t = self.time_array + (current_time % 10.0)
        vibration = (
            0.08 * np.sin(2 * np.pi * f1 * t) +
            0.04 * np.sin(2 * np.pi * f2 * t) +
            0.02 * np.sin(2 * np.pi * f3 * t) +
            0.015 * np.random.normal(0, 1, self.num_samples)
        )
        
        # If severe cracking, add impulsive harmonic micro-transients
        if crack_severity_ratio > 0.3:
            vibration += 0.05 * crack_severity_ratio * np.sin(2 * np.pi * (f1 * 3.5) * t)

        # FFT Analysis for Power Spectral Density
        fft_vals = np.abs(np.fft.rfft(vibration))
        fft_freqs = np.fft.rfftfreq(self.num_samples, 1.0 / self.sampling_rate)

        # Downsample for web transfer (max 60 points)
        downsample_step = max(1, len(fft_freqs) // 60)
        spectrum = [
            {"freq": round(float(fft_freqs[i]), 2), "amplitude": round(float(fft_vals[i]), 4)}
            for i in range(0, len(fft_freqs), downsample_step) if fft_freqs[i] <= 20.0
        ]

        # Time series signal sample (downsampled for waveform display)
        waveform_step = max(1, len(vibration) // 50)
        waveform = [
            round(float(vibration[i]), 4)
            for i in range(0, len(vibration), waveform_step)
        ]

        # Structural engineering sensor metrics
        peak_accel_g = round(float(np.max(np.abs(vibration))), 3)
        rms_accel_g = round(float(np.sqrt(np.mean(vibration ** 2))), 3)
        
        # Strain gauge simulation (micro-strain ue)
        base_strain = 220.0 + 45.0 * math.sin(current_time * 0.5)
        strain_ue = round(base_strain + 180.0 * crack_severity_ratio + np.random.uniform(-5, 5), 1)

        # Mid-span deflection LVDT (mm)
        base_deflection = 3.2 + 0.8 * math.sin(current_time * 0.3)
        deflection_mm = round(base_deflection + 4.5 * crack_severity_ratio + np.random.uniform(-0.1, 0.1), 2)

        # Pier tilt angle (arcmin)
        tilt_arcmin = round(0.4 + 1.2 * crack_severity_ratio + np.random.uniform(-0.02, 0.02), 3)

        # Environmental
        deck_temp_c = round(24.5 + 2.0 * math.sin(current_time * 0.05), 1)
        relative_humidity_pct = round(62.0 + 3.0 * math.cos(current_time * 0.05), 1)

        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "modal_frequencies_hz": {
                "mode_1_bending": round(f1, 2),
                "mode_2_bending": round(f2, 2),
                "mode_3_torsion": round(f3, 2),
                "frequency_drop_pct": round((1.0 - stiffness_factor) * 100.0, 1)
            },
            "sensors": {
                "peak_acceleration_g": peak_accel_g,
                "rms_acceleration_g": rms_accel_g,
                "microstrain_ue": strain_ue,
                "midspan_deflection_mm": deflection_mm,
                "pier_tilt_arcmin": tilt_arcmin,
                "deck_temp_c": deck_temp_c,
                "relative_humidity_pct": relative_humidity_pct
            },
            "fft_spectrum": spectrum,
            "waveform": waveform
        }

telemetry_simulator = BridgeTelemetrySimulator()
