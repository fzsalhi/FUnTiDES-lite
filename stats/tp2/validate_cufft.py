#!/usr/bin/env python3

import argparse
import csv
import sys

import numpy as np


def load_sismos(path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    names = data.dtype.names
    if names is None or len(names) < 2:
        raise RuntimeError("Invalid sismos CSV.")

    if names[0] != "time":
        raise RuntimeError("First column of sismos CSV must be 'time'.")

    time = np.asarray(data["time"], dtype=np.float64)
    if time.ndim == 0:
        time = np.array([time], dtype=np.float64)

    receiver_names = list(names[1:])
    signals = []

    for name in receiver_names:
        signal = np.asarray(data[name], dtype=np.float64)
        if signal.ndim == 0:
            signal = np.array([signal], dtype=np.float64)
        signals.append(signal)

    return time, receiver_names, signals


def load_cufft(path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    names = data.dtype.names
    expected = {"receiver", "frequency_hz", "amplitude"}

    if names is None or not expected.issubset(set(names)):
        raise RuntimeError(
            "FFT CSV must contain receiver,frequency_hz,amplitude."
        )

    receiver = np.atleast_1d(data["receiver"]).astype(int)
    frequency = np.atleast_1d(data["frequency_hz"]).astype(np.float64)
    amplitude = np.atleast_1d(data["amplitude"]).astype(np.float64)

    result = {}

    for r in np.unique(receiver):
        mask = receiver == r
        result[int(r)] = (
            frequency[mask],
            amplitude[mask],
        )

    return result


def dominant_frequency(freq, amp):
    if len(freq) <= 1:
        return float(freq[0])

    k = 1 + int(np.argmax(amp[1:]))
    return float(freq[k])


def main():
    parser = argparse.ArgumentParser(
        description="Compare NumPy ad-hoc FFT with in-situ cuFFT output."
    )
    parser.add_argument(
        "--sismos",
        default="sismos.csv",
        help="Input seismograms CSV."
    )
    parser.add_argument(
        "--cufft",
        default="fft_test.csv",
        help="In-situ cuFFT CSV."
    )
    parser.add_argument(
        "--csv-out",
        default="fft_validation.csv",
        help="Output validation summary CSV."
    )
    args = parser.parse_args()

    time, receiver_names, signals = load_sismos(args.sismos)
    cufft = load_cufft(args.cufft)

    if len(time) < 2:
        raise RuntimeError("At least two time samples are required.")

    dt_values = np.diff(time)
    dt = float(np.median(dt_values))

    if not np.allclose(dt_values, dt, rtol=1e-6, atol=1e-12):
        raise RuntimeError("Time samples are not uniformly spaced.")

    rows = []
    all_ok = True

    print(f"Samples   : {len(time)}")
    print(f"Receivers : {len(signals)}")
    print(f"dt        : {dt:.12g} s")
    print()

    for r, signal in enumerate(signals):
        if r not in cufft:
            print(f"[receiver {r}] missing in cuFFT CSV")
            all_ok = False
            continue

        centered = signal - np.mean(signal)

        numpy_fft = np.fft.rfft(centered)
        numpy_amp = np.abs(numpy_fft) * dt
        numpy_freq = np.fft.rfftfreq(len(signal), d=dt)

        cufft_freq, cufft_amp = cufft[r]

        if len(cufft_freq) != len(numpy_freq):
            print(
                f"[receiver {r}] size mismatch: "
                f"NumPy={len(numpy_freq)}, cuFFT={len(cufft_freq)}"
            )
            all_ok = False
            continue

        freq_error = float(np.max(np.abs(cufft_freq - numpy_freq)))
        freq_ok = np.allclose(
            cufft_freq,
            numpy_freq,
            rtol=1e-6,
            atol=1e-8
        )

        abs_error = np.abs(cufft_amp - numpy_amp)
        max_abs_error = float(np.max(abs_error))
        mean_abs_error = float(np.mean(abs_error))

        scale = np.maximum(np.abs(numpy_amp), 1e-12)
        max_rel_error = float(np.max(abs_error / scale))

        dom_numpy = dominant_frequency(numpy_freq, numpy_amp)
        dom_cufft = dominant_frequency(cufft_freq, cufft_amp)
        dom_error = abs(dom_numpy - dom_cufft)

        amp_ok = np.allclose(
            cufft_amp,
            numpy_amp,
            rtol=1e-4,
            atol=1e-6
        )

        receiver_ok = freq_ok and amp_ok
        all_ok = all_ok and receiver_ok

        rows.append({
            "receiver": r,
            "name": receiver_names[r],
            "n_samples": len(signal),
            "n_frequencies": len(numpy_freq),
            "max_frequency_error_hz": freq_error,
            "max_abs_amplitude_error": max_abs_error,
            "mean_abs_amplitude_error": mean_abs_error,
            "max_relative_amplitude_error": max_rel_error,
            "dominant_frequency_numpy_hz": dom_numpy,
            "dominant_frequency_cufft_hz": dom_cufft,
            "dominant_frequency_error_hz": dom_error,
            "status": "OK" if receiver_ok else "FAIL",
        })

        print(f"[receiver {r} - {receiver_names[r]}]")
        print(f"  max frequency error : {freq_error:.6e} Hz")
        print(f"  max amplitude error : {max_abs_error:.6e}")
        print(f"  mean amplitude error: {mean_abs_error:.6e}")
        print(f"  max relative error  : {max_rel_error:.6e}")
        print(f"  dominant NumPy      : {dom_numpy:.9g} Hz")
        print(f"  dominant cuFFT      : {dom_cufft:.9g} Hz")
        print(f"  status              : {'OK' if receiver_ok else 'FAIL'}")
        print()

    if rows:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Validation CSV written to: {args.csv_out}")

    print()
    print("GLOBAL STATUS:", "OK" if all_ok else "FAIL")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())