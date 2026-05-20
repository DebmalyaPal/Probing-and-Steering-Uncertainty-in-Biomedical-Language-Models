import os
from pathlib import Path

bioscope_dir = Path("data/bioscope")
openi_dir = Path("data/openi")

print(f"BioScope files: {len(list(bioscope_dir.rglob('*.xml')))}")
print(f"OpenI files: {len(list(openi_dir.rglob('*.xml')))}")