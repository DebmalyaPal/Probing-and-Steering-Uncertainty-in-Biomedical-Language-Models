import xml.etree.ElementTree as ET
from pathlib import Path
import random
import json

def extract_sentences_from_bioscope(xml_path):
    """Return (uncertain_sentences, certain_sentences) from a BioScope XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    uncertain = []
    certain = []

    for sentence in root.iter("sentence"):
        # Reconstruct full sentence text
        full_text = "".join(sentence.itertext()).strip()
        if not full_text or len(full_text) < 20:
            continue

        # Check if this sentence contains a speculation cue
        has_speculation = False
        for cue in sentence.iter("cue"):
            if cue.get("type") == "speculation":
                has_speculation = True
                break

        if has_speculation:
            uncertain.append(full_text)
        else:
            certain.append(full_text)

    return uncertain, certain


def build_balanced_contrast_set(bioscope_dir, max_per_class=200, seed=42):
    """Parse all BioScope files and return balanced uncertain/certain lists."""
    bioscope_dir = Path(bioscope_dir)
    xml_files = list(bioscope_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} BioScope XML files")

    all_uncertain = []
    all_certain = []

    for xf in xml_files:
        try:
            u, c = extract_sentences_from_bioscope(xf)
            all_uncertain.extend(u)
            all_certain.extend(c)
        except ET.ParseError as e:
            print(f"  Skipping {xf.name}: {e}")

    print(f"Extracted {len(all_uncertain)} uncertain, {len(all_certain)} certain")

    # Filter: keep sentences between 30 and 200 chars to control for length
    # This is important — length is a major confound in the vector
    all_uncertain = [s for s in all_uncertain if 30 <= len(s) <= 200]
    all_certain = [s for s in all_certain if 30 <= len(s) <= 200]
    print(f"After length filter: {len(all_uncertain)} uncertain, "
          f"{len(all_certain)} certain")

    # Balance and subsample
    random.seed(seed)
    n = min(len(all_uncertain), len(all_certain), max_per_class)
    uncertain_sample = random.sample(all_uncertain, n)
    certain_sample = random.sample(all_certain, n)

    return uncertain_sample, certain_sample


if __name__ == "__main__":
    u, c = build_balanced_contrast_set("data/bioscope",
                                        max_per_class=200)

    print(f"\nFinal contrast set: {len(u)} uncertain, {len(c)} certain")
    print("\nUncertain examples:")
    for s in u[:3]:
        print(f"  - {s[:150]}")
    print("\nCertain examples:")
    for s in c[:3]:
        print(f"  - {s[:150]}")

    # Save for reproducibility
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/bioscope_contrast.json", "w") as f:
        json.dump({
            "uncertain": u,
            "certain": c,
            "source": "BioScope abstracts",
            "length_filter": "30-200 chars",
            "seed": 42,
        }, f, indent=2)
    print("\nSaved to data/processed/bioscope_contrast.json")