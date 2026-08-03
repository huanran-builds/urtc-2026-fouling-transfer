RANDOM_STATE = 42
REDUCTION_THRESHOLD = 50.0          # percent, for outcome_class
SENSITIVITY_THRESHOLDS = [40.0, 60.0]
N_BOOTSTRAP = 1000
N_SPLITS = 5
DOMAINS = ["D1", "D2", "D3"]

CORE_DESCRIPTORS = [
    "water_contact_angle_deg", "surface_free_energy_mj_m2",
    "zeta_potential_mv", "rms_roughness_nm", "coating_thickness_nm",
    "elastic_modulus_mpa", "topography_feature_size_nm",
]
CATEGORICAL_DESCRIPTORS = ["charge_class", "coating_family"]
AGING_DESCRIPTORS = [
    "aging_exposure_hours", "aging_medium", "ion_release",
    "barrier_integrity_reported",
]
MIN_CORE_NONNULL = 3
