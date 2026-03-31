"""Backwards-compatible re-exports. Import from the package or submodules directly."""

from .model import (  # noqa: F401
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LABELS_PATH,
    NORMAL_LABEL_ALIASES,
    LoadedModel,
    PredictionResult,
    TransferLearningResNet50,
    choose_device,
    load_labels,
    load_model,
    predict_image,
    prepare_image,
    validate_model_artifacts,
)
from .gradcam import (  # noqa: F401
    generate_gradcam,
    save_gradcam,
)
from .artifacts import (  # noqa: F401
    ALLOWED_UPLOAD_SUFFIXES,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_UPLOADS_ROOT,
    analyze_saved_image,
    analyze_uploaded_panel,
    build_dashboard_result,
    create_run_id,
    ensure_unique_run_id,
    slugify,
)
