import os
import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# MICRO-MEND AI
# RGB + HSV MACHINE LEARNING MODEL
# ============================================================

DATASET_PATH = "micromend_color_dataset"

MODEL_OUTPUT = "micromend_color_ml_model.pkl"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "S1_yellow",
    "S2_yellow_green",
    "S3_green",
    "S4_blue_green",
    "S5_blue",
    "S6_purple"
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Convert BGR → HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Create color mask
    #
    # Ignore mostly white/gray background
    # --------------------------------------------------------

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    mask = (
        (saturation > 40) &
        (brightness > 40)
    )

    # If mask is too small, use entire image
    if np.sum(mask) < 100:

        rgb_pixels = rgb.reshape(-1, 3)
        hsv_pixels = hsv.reshape(-1, 3)

    else:

        rgb_pixels = rgb[mask]
        hsv_pixels = hsv[mask]

    # --------------------------------------------------------
    # RGB statistics
    # --------------------------------------------------------

    r = rgb_pixels[:, 0]
    g = rgb_pixels[:, 1]
    b = rgb_pixels[:, 2]

    # --------------------------------------------------------
    # HSV statistics
    # --------------------------------------------------------

    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]

    # --------------------------------------------------------
    # Create feature vector
    # --------------------------------------------------------

    features = [

        # RGB mean
        np.mean(r),
        np.mean(g),
        np.mean(b),

        # RGB standard deviation
        np.std(r),
        np.std(g),
        np.std(b),

        # RGB median
        np.median(r),
        np.median(g),
        np.median(b),

        # HSV mean
        np.mean(h),
        np.mean(s),
        np.mean(v),

        # HSV standard deviation
        np.std(h),
        np.std(s),
        np.std(v),

        # HSV median
        np.median(h),
        np.median(s),
        np.median(v),

        # HSV minimum
        np.min(h),
        np.min(s),
        np.min(v),

        # HSV maximum
        np.max(h),
        np.max(s),
        np.max(v),

    ]

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split):

    X = []
    y = []

    split_path = os.path.join(
        DATASET_PATH,
        split
    )

    print(
        f"\nLoading {split} dataset..."
    )

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.exists(class_path):

            print(
                f"WARNING: Missing folder: {class_path}"
            )

            continue

        files = os.listdir(class_path)

        image_count = 0

        for filename in files:

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            image_path = os.path.join(
                class_path,
                filename
            )

            features = extract_features(
                image_path
            )

            if features is not None:

                X.append(features)
                y.append(class_index)

                image_count += 1

        print(
            f"{class_name:<20} "
            f"{image_count} images"
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

X_train, y_train = load_dataset(
    "train"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

X_validation, y_validation = load_dataset(
    "validation"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test, y_test = load_dataset(
    "test"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n==========================================")
print("DATASET INFORMATION")
print("==========================================")

print(
    f"Training samples   : {len(X_train)}"
)

print(
    f"Validation samples : {len(X_validation)}"
)

print(
    f"Testing samples    : {len(X_test)}"
)

print(
    f"Number of features : {X_train.shape[1]}"
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n==========================================")
print("TRAINING RANDOM FOREST")
print("==========================================")

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    random_state=42,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train
)


# ============================================================
# TRAINING ACCURACY
# ============================================================

train_predictions = model.predict(
    X_train
)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)


# ============================================================
# VALIDATION ACCURACY
# ============================================================

validation_predictions = model.predict(
    X_validation
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)


# ============================================================
# TEST ACCURACY
# ============================================================

test_predictions = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print("MODEL RESULTS")
print("==========================================")

print(
    f"\nTraining Accuracy   : "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy : "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy       : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=CLASS_NAMES
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

matrix = confusion_matrix(
    y_test,
    test_predictions
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(matrix)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n==========================================")
print("TOP IMPORTANT FEATURES")
print("==========================================")

feature_names = [

    "R_mean",
    "G_mean",
    "B_mean",

    "R_std",
    "G_std",
    "B_std",

    "R_median",
    "G_median",
    "B_median",

    "H_mean",
    "S_mean",
    "V_mean",

    "H_std",
    "S_std",
    "V_std",

    "H_median",
    "S_median",
    "V_median",

    "H_min",
    "S_min",
    "V_min",

    "H_max",
    "S_max",
    "V_max"

]


importance = model.feature_importances_

indices = np.argsort(
    importance
)[::-1]


for i in indices[:10]:

    print(
        f"{feature_names[i]:<15} "
        f"{importance[i]:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_OUTPUT
)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    f"\nSaved as:"
    f"\n{MODEL_OUTPUT}"
)

print("\nMicro-Mend RGB + HSV model ready!")