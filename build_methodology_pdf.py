"""Build the two-page methodology handout without a PDF library dependency."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap


PAGES = [
    [
        "METHODLOGY AND WEIGHT PRIORS", "Beyond Curve Fitting: Virtual Sensors and Probabilistic Tyre Intelligence",
        "1. Decision objective", "The system estimates a hidden tyre degradation state from fuel-corrected green-flag lap times. It reports the probability that pace loss exceeds a 1.5 s cliff threshold, rather than only a point lap-time prediction.",
        "2. Public-data virtual sensors", "FastF1 does not provide private tyre CAN data. Track temperature, track profile and vehicle-dynamics priors create reproducible thermal-stress, pressure-deviation, slip-energy, lateral-load, brake-thermal and aero-load indices. Inputs and formulae are visible in tyre_features.py.",
        "3. State-space model", "State x = [degradation seconds, degradation-rate seconds/lap]. The transition advances state with tyre age. The observation is fuel-corrected degradation delta. A Kalman posterior supplies mean and covariance; innovation clipping limits traffic, lock-up and driver-error outliers.",
        "4. Uncertainty", "Forecasts propagate covariance into 70% (15th-85th) and 90% (5th-95th) fan-chart bands. Cliff probability is P(degradation > threshold), computed from the forecast normal distribution. Intervals are uncertainty estimates, not guarantees.",
    ],
    [
        "WEIGHT PRIOR RATIONALE AND VALIDATION", "5. Initial virtual-sensor prior weights", "Slip 0.16, thermal 0.16 and lateral 0.14 are highest because contact-patch shear and heat are primary wear mechanisms. Pressure 0.10, aero 0.09, brake and vertical load 0.08 modify operating conditions. Camber 0.06, differential 0.05, toe and brake bias 0.04 are lower-confidence setup modifiers.",
        "The weights are explicitly priors, not proprietary F1 measurements. update_wear_weight_priors applies only bounded, renormalised weekend evidence updates. This prevents a few noisy practice laps from overwriting the physical hierarchy.",
        "6. Validation protocol", "Hold out entire races, fit the global rate/noise prior on the remaining races, then score one-step-ahead pre-update predictions. For a race plot, overlay actual fuel-corrected degradation with the 70%/90% bands and report held-out interval coverage alongside MAE.",
        "7. References and limits", "H.B. Pacejka, Tire and Vehicle Dynamics, 3rd ed., Butterworth-Heinemann, 2012. TRICK tyre/road interaction work by Farroni et al. is the motivation for thermal and wear proxy modelling. The project brief also identifies recent FastF1 latent-state preprints; verify final bibliographic metadata before submission.",
        "This prototype never represents virtual sensors as measured competitor telemetry. Wet laps, safety-car laps, pit in/out laps and obvious slow-lap outliers are excluded from the base observations.",
    ],
]


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_page(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 17 Tf", "50 760 Td"]
    first = True
    for line in lines:
        heading = line[:2].isdigit() or line.startswith("WEIGHT")
        fragments = [line] if heading or line.startswith("METHOD") else wrap(line, width=92)
        for fragment in fragments:
            if not first:
                commands.append("0 -16 Td")
            commands.append("/F1 12 Tf" if heading else "/F1 10 Tf")
            commands.append(f"({escape(fragment)}) Tj")
            first = False
    commands.append("ET")
    return "\n".join(commands).encode("latin-1")


def write_pdf(destination: Path) -> None:
    streams = [build_page(page) for page in PAGES]
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>"]
    for n, stream in enumerate(streams):
        page_obj = 3 + n * 2
        content_obj = page_obj + 1
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents {content_obj} 0 R >>".encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    destination.write_bytes(output)


if __name__ == "__main__":
    write_pdf(Path("Methodology_and_Weights.pdf"))
