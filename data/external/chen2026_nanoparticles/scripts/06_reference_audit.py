import re
import pandas as pd

MIC = "data/external/chen2026_nanoparticles/Nanoparticles_MIC.csv"
MBC = "data/external/chen2026_nanoparticles/Nanoparticles_MBC.csv"

mic = pd.read_csv(MIC)
mbc = pd.read_csv(MBC)

mic["Ref"] = mic["Ref"].ffill()
mbc["Ref"] = mbc["Ref"].ffill()


def normalize_ref(ref):
    if pd.isna(ref):
        return None

    ref = str(ref).strip()

    # Extract the DOI itself and ignore surrounding junk
    match = re.search(
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
        ref,
        flags=re.I
    )

    if match:
        return match.group(0).rstrip(".,;").lower()

    return ref.lower()


mic["Ref_normalized"] = mic["Ref"].apply(normalize_ref)
mbc["Ref_normalized"] = mbc["Ref"].apply(normalize_ref)

mic_refs = set(mic["Ref_normalized"].dropna())
mbc_refs = set(mbc["Ref_normalized"].dropna())

print("MIC rows:", len(mic))
print("MIC unique normalized refs:", len(mic_refs))

print("MBC rows:", len(mbc))
print("MBC unique normalized refs:", len(mbc_refs))

print("References in BOTH:", len(mic_refs & mbc_refs))
print("MIC-only:", len(mic_refs - mbc_refs))
print("MBC-only:", len(mbc_refs - mic_refs))
print("Combined unique normalized refs:", len(mic_refs | mbc_refs))

print("\n--- REFS THAT COULD NOT BE PARSED AS DOI ---")

all_refs = pd.concat([
    mic[["Ref", "Ref_normalized"]],
    mbc[["Ref", "Ref_normalized"]]
]).drop_duplicates()

for _, row in all_refs.iterrows():
    if not str(row["Ref_normalized"]).startswith("10."):
        print(repr(row["Ref"]), "->", repr(row["Ref_normalized"]))

    