#!/usr/bin/env python3
"""Lecteur de fichiers .cpm (snapshots compresses) -> tableau numpy float.

Correspond a la struct C++ Header (40 octets) :
    uint32 magic | uint16 version | uint8 mode | uint8 nbits | uint8 flags |
    uint8[7] reserved | uint64 n_elements | uint64 payload_size |
    float scale | uint32 step
"""
import struct
import sys
from pathlib import Path

import numpy as np

# little-endian, sans padding : les offsets coincident avec ceux du C++
# (Q a l'offset 16, f a 32, I a 36 -> deja alignes naturellement)
HEADER_FMT = "<IHBBB7sQQfI"
HEADER_FIELDS = ("magic", "version", "mode", "nbits", "flags", "reserved",
                 "n_elements", "payload_size", "scale", "step")
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 40, f"en-tete de {HEADER_SIZE} octets, attendu 40"

# !! A verifier dans ton enum / tes constantes C++ !!
MODE_RAW, MODE_F16, MODE_INTN = 0, 1, 2
FLAG_DELTA, FLAG_RLE = 1, 2


def read_header(raw: bytes) -> dict:
    return dict(zip(HEADER_FIELDS, struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])))


def read_cpm(path, dtype_out=np.float32, return_header=False):
    """Lit un .cpm et renvoie les valeurs reconstruites (1D, ordre des noeuds)."""
    raw = Path(path).read_bytes()
    h = read_header(raw)
    payload = raw[HEADER_SIZE:HEADER_SIZE + h["payload_size"]]
    if len(payload) != h["payload_size"]:
        raise ValueError("fichier tronque : payload_size != octets presents")

    if h["flags"] & FLAG_RLE:
        raise NotImplementedError(
            "payload RLE : ajouter le decodeur (depend de ton format (valeur, compte))")

    n = h["n_elements"]
    mode = h["mode"]

    if mode == MODE_RAW:
        data = np.frombuffer(payload, dtype="<f4", count=n)
    elif mode == MODE_F16:
        data = np.frombuffer(payload, dtype="<f2", count=n)
    elif mode == MODE_INTN:
        nb = h["nbits"]
        if nb == 8:
            q = np.frombuffer(payload, dtype=np.int8, count=n)
        elif nb == 16:
            q = np.frombuffer(payload, dtype="<i2", count=n)
        elif 2 <= nb <= 16:
            # empaquete LSB-first, valeurs decalees de +qmax
            qmax = (1 << (nb - 1)) - 1
            bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8),
                                 bitorder="little")[: n * nb].reshape(n, nb)
            u = np.zeros(n, dtype=np.int32)
            for b in range(nb):            # boucle sur N bits, pas sur les noeuds
                u |= bits[:, b].astype(np.int32) << b
            q = u - qmax
        else:
            raise ValueError(f"nbits={nb} non gere")
        if h["flags"] & FLAG_DELTA:
            q = np.cumsum(q, dtype=q.dtype)   # decodage des differences
        data = q.astype(np.float64) * h["scale"]
    else:
        raise ValueError(f"mode inconnu : {mode}")

    data = data.astype(dtype_out)
    return (data, h) if return_header else data


def compare(path_cpm, ref):
    """RMSE et erreur max normalisees par max|p|, vs un tableau de reference."""
    rec = read_cpm(path_cpm, dtype_out=np.float64)
    ref = np.asarray(ref, dtype=np.float64).ravel()
    amax = np.max(np.abs(ref))
    err = rec - ref
    return {
        "rmse_norm": float(np.sqrt(np.mean(err**2)) / amax),
        "maxerr_norm": float(np.max(np.abs(err)) / amax),
    }


if __name__ == "__main__":
    for f in sys.argv[1:]:
        d, h = read_cpm(f, return_header=True)
        h.pop("reserved")
        print(f, h, "min/max =", d.min(), d.max())