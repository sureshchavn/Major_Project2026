import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\sures\.cache\kagglehub\datasets\andrewmvd\ocular-disease-recognition-odir5k\versions\2"

INPUT_CSV = os.path.join(DATASET_PATH, "odir5k_inspected.csv")

OUTPUT_DIR = r"C:\Users\sures\OneDrive\Desktop\Major-project\data"

TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")


# ============================================================
# DR STAGE MAPPING
# ============================================================

DR_STAGE_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


# ============================================================
# FIND IMAGE PATH
# ============================================================

def find_image(dataset_path, filename):
    """
    Search for the image inside the ODIR-5K dataset directory.
    """

    for root, dirs, files in os.walk(dataset_path):

        if filename in files:
            return os.path.join(root, filename)

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHASE 1.4 - PATIENT LEVEL DATASET PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input CSV
    # --------------------------------------------------------

    if not os.path.exists(INPUT_CSV):
        print("\nERROR: Input CSV not found!")
        print(INPUT_CSV)
        return

    print("\nInput CSV:")
    print(INPUT_CSV)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_CSV)

    print("\nOriginal dataset shape:")
    print(df.shape)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "ID",
        "Left-Fundus",
        "Right-Fundus",
        "left_dr_stage",
        "right_dr_stage"
    ]

    print("\nChecking required columns...")

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing columns:")
        print(missing_columns)

        print("\nAvailable columns:")
        print(df.columns.tolist())

        return

    print("All required columns found.")

    # --------------------------------------------------------
    # Create one row per eye
    # --------------------------------------------------------

    print("\nCreating eye-level dataset...")

    eye_records = []

    for _, row in df.iterrows():

        patient_id = int(row["ID"])

        # ====================================================
        # LEFT EYE
        # ====================================================

        left_stage = int(row["left_dr_stage"])

        if left_stage in DR_STAGE_NAMES:

            filename = str(row["Left-Fundus"])

            image_path = find_image(
                DATASET_PATH,
                filename
            )

            eye_records.append({
                "patient_id": patient_id,
                "eye": "left",
                "filename": filename,
                "image_path": image_path,
                "dr_stage": left_stage,
                "dr_label": DR_STAGE_NAMES[left_stage]
            })

        # ====================================================
        # RIGHT EYE
        # ====================================================

        right_stage = int(row["right_dr_stage"])

        if right_stage in DR_STAGE_NAMES:

            filename = str(row["Right-Fundus"])

            image_path = find_image(
                DATASET_PATH,
                filename
            )

            eye_records.append({
                "patient_id": patient_id,
                "eye": "right",
                "filename": filename,
                "image_path": image_path,
                "dr_stage": right_stage,
                "dr_label": DR_STAGE_NAMES[right_stage]
            })

    eye_df = pd.DataFrame(eye_records)

    print("\nEye-level dataset shape:")
    print(eye_df.shape)

    # --------------------------------------------------------
    # Check image paths
    # --------------------------------------------------------

    print("\nChecking image files...")

    missing_images = eye_df["image_path"].isna().sum()

    print("Missing images:", missing_images)

    if missing_images > 0:

        print("\nRemoving records with missing images...")

        eye_df = eye_df.dropna(
            subset=["image_path"]
        ).reset_index(drop=True)

    print(
        "Records after image verification:",
        len(eye_df)
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before_duplicates = len(eye_df)

    eye_df = eye_df.drop_duplicates(
        subset=["patient_id", "eye", "filename"]
    ).reset_index(drop=True)

    after_duplicates = len(eye_df)

    print("\nDuplicate records removed:",
          before_duplicates - after_duplicates)

    # --------------------------------------------------------
    # Patient-level split
    # --------------------------------------------------------

    print("\nCreating patient-level split...")

    patients = eye_df["patient_id"].unique()

    print("Total unique patients:", len(patients))

    # 70% Train, 15% Validation, 15% Test
    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=42
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=42
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_df = eye_df[
        eye_df["patient_id"].isin(train_patients)
    ].copy()

    val_df = eye_df[
        eye_df["patient_id"].isin(val_patients)
    ].copy()

    test_df = eye_df[
        eye_df["patient_id"].isin(test_patients)
    ].copy()

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_CSV,
        index=False
    )

    val_df.to_csv(
        VAL_CSV,
        index=False
    )

    test_df.to_csv(
        TEST_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Print dataset sizes
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print("\nTrain:")
    print("Images:", len(train_df))
    print("Patients:", train_df["patient_id"].nunique())

    print("\nValidation:")
    print("Images:", len(val_df))
    print("Patients:", val_df["patient_id"].nunique())

    print("\nTest:")
    print("Images:", len(test_df))
    print("Patients:", test_df["patient_id"].nunique())

    # --------------------------------------------------------
    # Verify patient leakage
    # --------------------------------------------------------

    train_set = set(train_df["patient_id"])
    val_set = set(val_df["patient_id"])
    test_set = set(test_df["patient_id"])

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    print("\n" + "=" * 60)
    print("PATIENT LEAKAGE CHECK")
    print("=" * 60)

    print("Train ∩ Validation:",
          len(train_val_overlap))

    print("Train ∩ Test:",
          len(train_test_overlap))

    print("Validation ∩ Test:",
          len(val_test_overlap))

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    for name, dataset in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df)
    ]:

        print("\n" + name)

        counts = dataset["dr_stage"].value_counts().sort_index()

        for stage in range(5):

            count = counts.get(stage, 0)

            print(
                f"{stage} - "
                f"{DR_STAGE_NAMES[stage]}: "
                f"{count}"
            )

    # --------------------------------------------------------
    # Final file locations
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print("\nTrain:")
    print(TRAIN_CSV)

    print("\nValidation:")
    print(VAL_CSV)

    print("\nTest:")
    print(TEST_CSV)

    print("\n" + "=" * 60)
    print("PHASE 1.4 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()