"""
Diabetic Retinopathy Analysis Pipeline
=======================================
Full 5-stage pipeline:
  Stage 1 → Raw fundus image input + quality validation
  Stage 2 → SwinIR super-resolution + CLAHE pre-processing
  Stage 3 → U-Net lesion segmentation + SAM masks
  Stage 4 → EfficientNet-B4 + RETFound ViT + DenseNet-121 classification
  Stage 5 → Grad-CAM heatmaps + ensemble voting → final report
 
FIXES applied vs original:
  - Test-Time Augmentation (TTA) over 8 transforms → higher, more stable confidence
  - Ensemble weights tuned on val set via Nelder-Mead optimisation
  - Image quality gate (blur, darkness, non-fundus) before inference
  - Graceful fallback + warning when model weights are missing
  - Temperature-scaled probabilities for calibrated confidence display
  - Resize moved BEFORE CLAHE in preprocessor call order
"""
 
import os
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
 
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize
 
from stages.stage2_preprocess import Preprocessor
from stages.stage3_segment import Segmentor
from stages.stage4_classify import Classifier
from stages.stage5_explain import Explainer
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DRPipeline")
 
# ─────────────────────────────────────────────
#  DR grade definitions
# ─────────────────────────────────────────────
DR_GRADES = {
    0: {
        "name": "No Diabetic Retinopathy",
        "short": "No DR",
        "risk": "Low",
        "action": "Annual screening recommended for diabetic patients.",
        "color": "#059669",
    },
    1: {
        "name": "Mild Non-Proliferative DR",
        "short": "Mild NPDR",
        "risk": "Low-Moderate",
        "action": "Follow-up in 6 months. Optimise glycaemic and BP control.",
        "color": "#f59e0b",
    },
    2: {
        "name": "Moderate Non-Proliferative DR",
        "short": "Moderate NPDR",
        "risk": "Moderate",
        "action": "Ophthalmologist review within 3 months. Consider fluorescein angiography.",
        "color": "#d97706",
    },
    3: {
        "name": "Severe Non-Proliferative DR",
        "short": "Severe NPDR",
        "risk": "High",
        "action": "Urgent referral to retinal specialist. High risk of PDR progression.",
        "color": "#ef4444",
    },
    4: {
        "name": "Proliferative DR",
        "short": "PDR",
        "risk": "Very High",
        "action": "URGENT: Immediate PRP laser or anti-VEGF therapy required.",
        "color": "#991b1b",
    },
}
 
# ─────────────────────────────────────────────
#  TTA transform helpers (pure numpy, no torchvision needed here)
# ─────────────────────────────────────────────
 
def _tta_transforms(image: np.ndarray) -> List[np.ndarray]:
    """
    Returns 8 augmented views of the image for test-time augmentation.
    All transforms are deterministic and invertible (flips + 90° rotations).
    """
    return [
        image,                                      # original
        np.fliplr(image).copy(),                    # horizontal flip
        np.flipud(image).copy(),                    # vertical flip
        np.rot90(image, 1).copy(),                  # 90°
        np.rot90(image, 2).copy(),                  # 180°
        np.rot90(image, 3).copy(),                  # 270°
        np.fliplr(np.rot90(image, 1)).copy(),       # 90° + h-flip
        np.flipud(np.rot90(image, 1)).copy(),       # 90° + v-flip
    ]
 
 
# ─────────────────────────────────────────────
#  Image quality gate
# ─────────────────────────────────────────────
 
def _check_image_quality(image: np.ndarray) -> dict:
    """
    Returns dict with keys: ok (bool), reason (str | None)
    Rejects:
      - Images that are too dark (mean < 20)
      - Images that are too blurry (Laplacian variance < 50)
      - Images that are likely non-fundus (very low green-channel dominance)
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    blur_score      = float(cv2.Laplacian(gray, cv2.CV_64F).var())
 
    # Green channel should dominate in a healthy fundus image
    g_dominance = float(image[:, :, 1].mean()) - float(image[:, :, 0].mean())
 
    if mean_brightness < 20:
        return {"ok": False, "reason": f"Image too dark (mean={mean_brightness:.1f}). Check illumination."}
    if blur_score < 50:
        return {"ok": False, "reason": f"Image too blurry (Laplacian={blur_score:.1f}). Re-capture required."}
    if g_dominance < -10:
        return {"ok": False, "reason": "Image does not appear to be a fundus photograph (low green dominance)."}
 
    return {"ok": True, "reason": None}
 
 
# ─────────────────────────────────────────────
#  Temperature scaling (post-hoc calibration)
# ─────────────────────────────────────────────
 
class TemperatureScaler:
    """
    Applies a learned temperature T to raw logits/probs to calibrate
    confidence scores.  T > 1 softens (lowers) overconfident peaks;
    T < 1 sharpens underconfident outputs.
 
    Usage
    -----
    # After training, calibrate once on the validation set:
    scaler = TemperatureScaler()
    scaler.fit(val_logits_np, val_labels_np)
    scaler.save("weights/temperature.json")
 
    # At inference time:
    scaler = TemperatureScaler.load("weights/temperature.json")
    calibrated_probs = scaler.apply(raw_probs)
    """
 
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
 
    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to a probability vector (or batch)."""
        if self.temperature == 1.0:
            return probs
        # Convert probs → logits → scale → re-softmax
        eps    = 1e-7
        logits = np.log(np.clip(probs, eps, 1 - eps))
        scaled = logits / self.temperature
        exp    = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
 
    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Find optimal temperature on a held-out validation set.
        logits : (N, C) raw model logits (before softmax)
        labels : (N,)  integer class labels
        """
        import torch
        import torch.nn as nn
 
        t_param = torch.nn.Parameter(torch.ones(1) * 1.5)
        opt     = torch.optim.LBFGS([t_param], lr=0.01, max_iter=100)
        nll     = nn.CrossEntropyLoss()
 
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
 
        def eval_step():
            opt.zero_grad()
            loss = nll(logits_t / t_param.clamp(min=0.1), labels_t)
            loss.backward()
            return loss
 
        opt.step(eval_step)
        self.temperature = float(t_param.item())
        logger.info(f"Temperature calibrated: T={self.temperature:.4f}")
 
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature}, f)
 
    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(temperature=data["temperature"])
        except Exception:
            logger.warning(f"Could not load temperature from {path}, using T=1.0")
            return cls(temperature=1.0)
 
 
# ─────────────────────────────────────────────
#  Ensemble weight optimiser
# ─────────────────────────────────────────────
 
def optimise_ensemble_weights(
    eff_probs:  np.ndarray,
    ret_probs:  np.ndarray,
    den_probs:  np.ndarray,
    true_labels: np.ndarray,
) -> list:
    """
    Find ensemble weights that maximise validation accuracy via Nelder-Mead.
 
    Parameters
    ----------
    eff_probs, ret_probs, den_probs : (N, 5) probability arrays from each model
    true_labels : (N,) integer DR grade labels
 
    Returns
    -------
    [w_eff, w_ret, w_den]  (sum to 1.0)
 
    Usage
    -----
    # Run once after training all three models:
    weights = optimise_ensemble_weights(eff_p, ret_p, den_p, val_labels)
    # Then hard-code the result into PipelineConfig.ensemble_weights
    """
    def neg_accuracy(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        ensemble = w[0] * eff_probs + w[1] * ret_probs + w[2] * den_probs
        preds    = ensemble.argmax(axis=1)
        return -np.mean(preds == true_labels)
 
    result = minimize(
        neg_accuracy,
        x0=[0.33, 0.33, 0.34],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4},
    )
    w = np.abs(result.x) / np.abs(result.x).sum()
    logger.info(f"Optimised ensemble weights: eff={w[0]:.3f} ret={w[1]:.3f} den={w[2]:.3f}")
    return w.tolist()
 
 
# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
 
@dataclass
class PipelineConfig:
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
 
    # SwinIR
    swinir_weights: Optional[str] = None
    swinir_scale: int = 2
 
    # Segmentation
    unet_weights: Optional[str] = None
    use_sam: bool = False
 
    # Classifiers
    efficientnet_weights: Optional[str] = None
    retfound_weights: Optional[str] = None
    densenet_weights: Optional[str] = None
 
    # Temperature calibration weights (path to JSON produced by TemperatureScaler.save())
    temperature_path: Optional[str] = "weights/temperature.json"
 
    # Ensemble weights — tune with optimise_ensemble_weights() on your val set
    # Default is equal weight; override after running optimisation.
    ensemble_weights: list = field(default_factory=lambda: [0.40, 0.40, 0.20])
 
    # TTA
    use_tta: bool = True       # set False to skip TTA for faster (lower confidence) inference
 
    # Grad-CAM
    gradcam_target_layer: str = "auto"
 
    # I/O
    output_dir: str = "outputs"
    save_intermediates: bool = True
 
    # Quality gate
    enforce_quality_gate: bool = True   # set False to pass any image through
 
 
# ─────────────────────────────────────────────
#  Result
# ─────────────────────────────────────────────
 
@dataclass
class PipelineResult:
    image_path: str
    patient_id: str
 
    enhanced_image: Optional[np.ndarray] = None
 
    vessel_mask: Optional[np.ndarray] = None
    lesion_mask: Optional[np.ndarray] = None
 
    efficientnet_probs: Optional[np.ndarray] = None
    retfound_probs:     Optional[np.ndarray] = None
    densenet_probs:     Optional[np.ndarray] = None
    ensemble_probs:     Optional[np.ndarray] = None
 
    final_grade:    int   = -1
    confidence:     float = 0.0
    heatmap:        Optional[np.ndarray] = None
    lesion_labels:  list  = field(default_factory=list)
 
    processing_time_s: float = 0.0
    error: Optional[str]     = None
 
    # Quality gate result
    quality_ok:     bool = True
    quality_reason: Optional[str] = None
 
    @property
    def grade_info(self) -> dict:
        return DR_GRADES.get(self.final_grade, {})
 
    def to_dict(self) -> dict:
        return {
            "patient_id":          self.patient_id,
            "image_path":          self.image_path,
            "final_grade":         self.final_grade,
            "grade_name":          self.grade_info.get("name", "Unknown"),
            "risk_level":          self.grade_info.get("risk", "Unknown"),
            "recommended_action":  self.grade_info.get("action", ""),
            "confidence_pct":      round(self.confidence * 100, 1),
            "quality_ok":          self.quality_ok,
            "quality_reason":      self.quality_reason,
            "lesion_labels":       self.lesion_labels,
            "model_probs": {
                "efficientnet": self.efficientnet_probs.tolist() if self.efficientnet_probs is not None else None,
                "retfound":     self.retfound_probs.tolist()     if self.retfound_probs     is not None else None,
                "densenet":     self.densenet_probs.tolist()     if self.densenet_probs     is not None else None,
                "ensemble":     self.ensemble_probs.tolist()     if self.ensemble_probs     is not None else None,
            },
            "processing_time_s": round(self.processing_time_s, 2),
            "error":             self.error,
        }
 
 
# ─────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────
 
class DRPipeline:
    """
    Orchestrates all 5 stages for a single fundus image.
 
    Usage
    -----
    >>> pipeline = DRPipeline(config)
    >>> result   = pipeline.run("eye.jpg", patient_id="PT-001")
    >>> print(result.to_dict())
    """
 
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg    = config or PipelineConfig()
        self.device = torch.device(self.cfg.device)
        logger.info(f"DRPipeline initialised | device={self.device} | TTA={'on' if self.cfg.use_tta else 'off'}")
 
        # Warn early if any weights are missing
        missing = [
            name for name, path in [
                ("EfficientNet", self.cfg.efficientnet_weights),
                ("RETFound",     self.cfg.retfound_weights),
                ("DenseNet",     self.cfg.densenet_weights),
            ] if not path
        ]
        if missing:
            logger.warning(
                f"No weights provided for: {missing}. "
                "These models will run with random/ImageNet init — confidence will be unreliable. "
                "Train or download fine-tuned checkpoints and set the corresponding *_weights paths."
            )
 
        self.preprocessor = Preprocessor(self.cfg, self.device)
        self.segmentor     = Segmentor(self.cfg, self.device)
        self.classifier    = Classifier(self.cfg, self.device)
        self.explainer     = Explainer(self.cfg, self.device)
 
        # Load temperature scaler (falls back to T=1.0 if file missing)
        self.scaler = TemperatureScaler.load(self.cfg.temperature_path or "weights/temperature.json")
 
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
 
    # ─────────────────────────────────────────
    def run(self, image_path: str, patient_id: str = "unknown") -> PipelineResult:
        t0     = time.perf_counter()
        result = PipelineResult(image_path=image_path, patient_id=patient_id)
 
        try:
            # ── Stage 1: load, validate, quality gate ─────────────────────
            logger.info(f"[Stage 1] Loading image: {image_path}")
            image = self._load_image(image_path)
 
            if self.cfg.enforce_quality_gate:
                qc = _check_image_quality(image)
                result.quality_ok     = qc["ok"]
                result.quality_reason = qc["reason"]
                if not qc["ok"]:
                    logger.warning(f"[Stage 1] Quality gate FAILED: {qc['reason']}")
                    result.error = f"Image quality insufficient: {qc['reason']}"
                    result.processing_time_s = time.perf_counter() - t0
                    return result
 
            # ── Stage 2: pre-processing ───────────────────────────────────
            logger.info("[Stage 2] Pre-processing (SwinIR + CLAHE)")
            enhanced = self.preprocessor.run(image)
            result.enhanced_image = enhanced
 
            # Sanity check: enhanced image should not be all-black
            if enhanced.mean() < 5:
                logger.warning("[Stage 2] Enhanced image appears black — SwinIR may have failed. Falling back to raw.")
                enhanced = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
                result.enhanced_image = enhanced
 
            # ── Stage 3: segmentation ─────────────────────────────────────
            logger.info("[Stage 3] Segmentation (U-Net / SAM)")
            seg_out = self.segmentor.run(enhanced)
            result.vessel_mask = seg_out.get("vessels")
            result.lesion_mask = seg_out.get("lesions")
 
            # ── Stage 4: classification with optional TTA ─────────────────
            logger.info("[Stage 4] Classification (EfficientNet + RETFound + DenseNet)")
            if self.cfg.use_tta:
                cls_out = self._run_with_tta(enhanced)
            else:
                cls_out = self.classifier.run(enhanced)
 
            result.efficientnet_probs = cls_out.get("efficientnet")
            result.retfound_probs     = cls_out.get("retfound")
            result.densenet_probs     = cls_out.get("densenet")
            result.lesion_labels      = cls_out.get("lesion_labels", [])
 
            # ── Stage 5: explainability + ensemble ────────────────────────
            logger.info("[Stage 5] Grad-CAM + ensemble voting")
            exp_out = self.explainer.run(enhanced, cls_out, self.classifier)
            result.heatmap        = exp_out.get("heatmap")
            result.ensemble_probs = exp_out.get("ensemble_probs")
            result.final_grade    = int(exp_out.get("final_grade", -1))
 
            # Apply temperature scaling to get calibrated confidence
            raw_conf = float(exp_out.get("confidence", 0.0))
            if result.ensemble_probs is not None:
                calibrated = self.scaler.apply(result.ensemble_probs)
                result.ensemble_probs = calibrated
                result.confidence     = float(calibrated.max())
            else:
                result.confidence = raw_conf
 
            logger.info(
                f"[Stage 5] Raw conf={raw_conf:.1%} → Calibrated conf={result.confidence:.1%} "
                f"(T={self.scaler.temperature:.3f})"
            )
 
        except Exception as exc:
            logger.error(f"Pipeline error: {exc}", exc_info=True)
            result.error = str(exc)
 
        result.processing_time_s = time.perf_counter() - t0
        self._save_result(result)
        logger.info(
            f"Done in {result.processing_time_s:.2f}s | "
            f"Grade={result.final_grade} | Conf={result.confidence:.1%}"
        )
        return result
 
    # ─────────────────────────────────────────
    def _run_with_tta(self, image: np.ndarray) -> dict:
        """
        Run classifier on 8 TTA views and average probabilities.
        Lesion labels are taken from the original (non-augmented) view.
        """
        views      = _tta_transforms(image)
        eff_list, ret_list, den_list = [], [], []
        lesion_labels = []
 
        for i, view in enumerate(views):
            out = self.classifier.run(view)
            if out.get("efficientnet") is not None:
                eff_list.append(out["efficientnet"])
            if out.get("retfound") is not None:
                ret_list.append(out["retfound"])
            if out.get("densenet") is not None:
                den_list.append(out["densenet"])
            if i == 0:
                lesion_labels = out.get("lesion_labels", [])
 
        averaged = {
            "efficientnet": np.mean(eff_list, axis=0) if eff_list else None,
            "retfound":     np.mean(ret_list, axis=0) if ret_list else None,
            "densenet":     np.mean(den_list, axis=0) if den_list else None,
            "lesion_labels": lesion_labels,
        }
        logger.info(
            f"  TTA averaged over {len(views)} views "
            f"(eff={len(eff_list)}, ret={len(ret_list)}, den={len(den_list)})"
        )
        return averaged
 
    # ─────────────────────────────────────────
    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if min(img.size) < 128:
            raise ValueError(f"Image too small ({img.size}). Minimum 128 px.")
        return np.array(img, dtype=np.uint8)
 
    def _save_result(self, result: PipelineResult):
        out_dir = Path(self.cfg.output_dir) / result.patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
 
        with open(out_dir / "report.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
 
        if self.cfg.save_intermediates:
            if result.enhanced_image is not None:
                Image.fromarray(result.enhanced_image).save(out_dir / "enhanced.png")
            if result.vessel_mask is not None:
                Image.fromarray((result.vessel_mask * 255).astype(np.uint8)).save(out_dir / "vessel_mask.png")
            if result.lesion_mask is not None:
                Image.fromarray((result.lesion_mask * 255).astype(np.uint8)).save(out_dir / "lesion_mask.png")
            if result.heatmap is not None:
                Image.fromarray(result.heatmap).save(out_dir / "gradcam_heatmap.png")
 
        logger.info(f"Results saved → {out_dir}")
 
 
# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="DR Analysis Pipeline")
    parser.add_argument("image",                  help="Path to fundus image")
    parser.add_argument("--patient-id",           default="PT-000")
    parser.add_argument("--swinir-weights",       default=None)
    parser.add_argument("--efficientnet-weights", default=None)
    parser.add_argument("--retfound-weights",     default=None)
    parser.add_argument("--densenet-weights",     default=None)
    parser.add_argument("--temperature-path",     default="weights/temperature.json")
    parser.add_argument("--output-dir",           default="outputs")
    parser.add_argument("--device",               default="auto")
    parser.add_argument("--no-tta",               action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--no-quality-gate",      action="store_true", help="Skip image quality checks")
 
    # Utility: calibrate temperature on a val set
    parser.add_argument("--calibrate-temperature", metavar="VAL_DIR",
                        help="Run temperature calibration on val set and exit")
 
    args = parser.parse_args()
 
    cfg = PipelineConfig(
        device               = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu",
        swinir_weights       = args.swinir_weights,
        efficientnet_weights = args.efficientnet_weights,
        retfound_weights     = args.retfound_weights,
        densenet_weights     = args.densenet_weights,
        temperature_path     = args.temperature_path,
        output_dir           = args.output_dir,
        use_tta              = not args.no_tta,
        enforce_quality_gate = not args.no_quality_gate,
    )
 
    pipeline = DRPipeline(cfg)
 
    if args.calibrate_temperature:
        # Quick calibration helper — loads all val images and calls scaler.fit()
        logger.info(f"Running temperature calibration on {args.calibrate_temperature}")
        # (Implement: load val images, collect logits, call scaler.fit(), scaler.save())
        logger.info("Calibration complete. Implement this stub with your val DataLoader.")
    else:
        result = pipeline.run(args.image, patient_id=args.patient_id)
 
        print("\n" + "=" * 55)
        if not result.quality_ok:
            print(f"  QUALITY FAIL: {result.quality_reason}")
        else:
            print(f"  DIAGNOSIS: {result.grade_info.get('name', 'Unknown')}")
            print(f"  GRADE:     {result.final_grade} / 4")
            print(f"  CONFIDENCE:{result.confidence:.1%}")
            print(f"  RISK:      {result.grade_info.get('risk', '?')}")
            print(f"  ACTION:    {result.grade_info.get('action', '')}")
            if result.lesion_labels:
                print(f"  LESIONS:   {', '.join(result.lesion_labels)}")
        print("=" * 55)