GENDER = {"Female": 0, "Male": 1}
BINARY = {"No": 0, "Yes": 1}

MULTIPLE_LINES = {
    "No": 0,
    "No phone service": 1,
    "Yes": 2,
}

INTERNET_SERVICE = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2,
}

INTERNET_ADDON = {
    "No": 0,
    "No internet service": 1,
    "Yes": 2,
}

CONTRACT = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}

PAYMENT = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

FEATURE_ENCODINGS = {
    "gender": GENDER,
    "Partner": BINARY,
    "Dependents": BINARY,
    "PhoneService": BINARY,
    "MultipleLines": MULTIPLE_LINES,
    "InternetService": INTERNET_SERVICE,
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": CONTRACT,
    "PaperlessBilling": BINARY,
    "PaymentMethod": PAYMENT,
}


def encode_category(feature: str, value: str) -> int:
    try:
        return FEATURE_ENCODINGS[feature][value]
    except KeyError as exc:
        raise ValueError(f"Unknown value {value!r} for feature {feature!r}") from exc
