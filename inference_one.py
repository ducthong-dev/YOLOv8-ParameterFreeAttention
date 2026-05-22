import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# --- CONFIG ---
DEVICE = "mps"  # or 'cuda' or 'cpu'
IMG_PATH = "dataset/Plant_leaf_diseases_dataset/augmented_test_images_hardest/Apple___Apple_scab/img_44.png"
ORIG_MODEL_PATH = (
    "models/drive-download-20250529T055242Z-1-001/220525-YOLOv8n_cls/weights/best.pt"
)
ECA_MODEL_PATH = "models/drive-download-20250529T055242Z-1-001/230525-YOLOv8n_cls_ECA/weights/last.pt"

# --- LOAD IMAGE ---
img = Image.open(IMG_PATH).convert("RGB").resize((224, 224))
img_np = np.array(img) / 255.0
img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


# --- UTILITY FUNCTIONS ---
def register_target_layer(model, feature_dict, classifier="Classify", target="Conv"):
    found = False
    for m in reversed(list(model.model.modules())):
        if m.__class__.__name__ == classifier:
            found = True
        elif found and m.__class__.__name__ == target:
            m.register_forward_hook(
                lambda _, __, output: feature_dict.update(
                    {"map": output.detach().cpu()}
                )
            )
            return m
    raise ValueError(f"Layer {target} before {classifier} not found.")


def compute_gradcam(model, img_tensor, module, class_idx):
    activations, gradients = {}, {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fwd = module.register_forward_hook(forward_hook)
    bwd = module.register_full_backward_hook(backward_hook)

    img_tensor.requires_grad = True
    pred = model.model(img_tensor)
    pred_score = pred[0, class_idx]
    model.model.zero_grad()
    pred_score.backward(retain_graph=True)

    act, grad = activations["value"][0], gradients["value"][0]
    weights = grad.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    fwd.remove()
    bwd.remove()
    return cam.cpu().numpy()


def overlay_cam(img, cam, alpha=0.5):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        img.size, resample=Image.BILINEAR
    )
    heatmap = plt.get_cmap("jet")(np.array(cam_img) / 255.0)[:, :, :3]
    overlay = np.clip(np.array(img) / 255.0 * (1 - alpha) + heatmap * alpha, 0, 1)
    return overlay


def plot_feature_maps_and_histograms(fmap1, fmap2, label1="YOLOv8n", label2="ECA"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].imshow(fmap1[-1], cmap="viridis")
    axs[0, 0].set_title(f"{label1} Feature Map (Last Channel)")
    axs[0, 1].imshow(fmap2[-1], cmap="viridis")
    axs[0, 1].set_title(f"{label2} Feature Map (Last Channel)")
    axs[1, 0].hist(fmap1.flatten(), bins=50, color="blue", alpha=0.7)
    axs[1, 0].set_title(f"{label1} Activation Histogram")
    axs[1, 1].hist(fmap2.flatten(), bins=50, color="orange", alpha=0.7)
    axs[1, 1].set_title(f"{label2} Activation Histogram")
    for ax in axs.ravel():
        ax.axis("off") if len(ax.images) > 0 else None
    plt.tight_layout()
    plt.show()

    # Print the total elements of the histograms
    total_elements_fmap1 = fmap1.size
    total_elements_fmap2 = fmap2.size
    print(f"Total elements in {label1} histogram: {total_elements_fmap1}")
    print(f"Total elements in {label2} histogram: {total_elements_fmap2}")


# --- COMPARE PRE/POST ECA FEATURES ---
def compare_pre_post_eca_features(model, img_tensor):
    fmap_before, fmap_after = {}
    for m in reversed(list(model.model.modules())):
        if m.__class__.__name__ == "Classify":
            break
        if m.__class__.__name__ == "Conv":
            m.register_forward_hook(
                lambda _, __, output: fmap_before.setdefault(
                    "value", output.detach().cpu()
                )
            )
            break
    for m in model.model.modules():
        if m.__class__.__name__.lower().startswith("eca"):
            m.register_forward_hook(
                lambda _, __, output: fmap_after.setdefault(
                    "value", output.detach().cpu()
                )
            )
            break
    _ = model.model(img_tensor)
    if "value" not in fmap_before or "value" not in fmap_after:
        raise ValueError("Không hook được layer trước hoặc sau ECA.")
    return fmap_before["value"][0].numpy(), fmap_after["value"][0].numpy()


def plot_gradcam_comparison(img, cam1, cam2, label1="YOLOv8n", label2="ECA"):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(overlay_cam(img, cam1))
    axs[1].set_title(f"Grad-CAM: {label1}")
    axs[1].axis("off")
    axs[2].imshow(overlay_cam(img, cam2))
    axs[2].set_title(f"Grad-CAM: {label2}")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()


def plot_feature_map_comparison(fmap1, fmap2, label1="YOLOv8n", label2="ECA"):
    plot_feature_maps_and_histograms(fmap1, fmap2, label1, label2)


# --- INFERENCE & HOOK ---
# Debugging model loading
print(f"Loading original model from: {ORIG_MODEL_PATH}")
print(f"Loading ECA model from: {ECA_MODEL_PATH}")

try:
    orig_model = YOLO(ORIG_MODEL_PATH)
    if orig_model.model is None:
        raise ValueError(
            "Failed to load the original YOLO model. Check the model path or weights."
        )
except Exception as e:
    raise ValueError(f"Error loading original model: {e}")

try:
    eca_model = YOLO(ECA_MODEL_PATH)
    if eca_model.model is None:
        raise ValueError(
            "Failed to load the ECA YOLO model. Check the model path or weights."
        )
except Exception as e:
    raise ValueError(f"Error loading ECA model: {e}")

orig_feat, eca_feat = {}, {}
orig_layer = register_target_layer(orig_model, orig_feat)
eca_layer = register_target_layer(eca_model, eca_feat)

with torch.no_grad():
    orig_logits = orig_model.model(img_tensor)
    eca_logits = eca_model.model(img_tensor)
orig_pred = torch.argmax(orig_logits, dim=1).item()
eca_pred = torch.argmax(eca_logits, dim=1).item()

# --- GRAD-CAM COMPUTATION ---
orig_cam = compute_gradcam(orig_model, img_tensor.clone(), orig_layer, orig_pred)
eca_cam = compute_gradcam(eca_model, img_tensor.clone(), eca_layer, eca_pred)

plot_gradcam_comparison(img, orig_cam, eca_cam)
plot_feature_map_comparison(orig_feat["map"][0].numpy(), eca_feat["map"][0].numpy())

# --- PREDICTION SUMMARY ---
print(f"YOLOv8n Prediction: {orig_pred}")
print(f"ECA Model Prediction: {eca_pred}")
