import base64
import numpy as np
import cv2
from PIL import Image
import io
import os
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------- UNet model definition (same as before) ---------
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.middle = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# ---------- Global Model ----------
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load Model ----------
def load_segmentation_model():
    global model
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'unet_segmentation.pth')

        if not os.path.exists(model_path):
            logger.error(f" UNet model file not found at: {model_path}")
            return False

        logger.info(f" Loading UNet model from: {model_path}")

        model = UNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        logger.info(" UNet segmentation model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f" Failed to load UNet model: {str(e)}")
        return False


# ---------- Run Inference ----------
def run_segmentation_with_masks(image_data):
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_resized = image.resize((256, 256))

        transform = transforms.ToTensor()
        input_tensor = transform(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        original = np.array(image_resized)
        mask_3d = np.stack([pred_mask]*3, axis=-1)

        # Light red overlay
        colored_mask = np.zeros_like(original)
        colored_mask[:, :, 0] = 255
        colored_mask[:, :, 1] = 100
        colored_mask[:, :, 2] = 100

        alpha = 0.4
        overlayed = np.where(mask_3d == 1,
                            (alpha * colored_mask + (1 - alpha) * original).astype(np.uint8),
                            original)

        detections = [{
            "class": 0,
            "name": "segmentation"
        }] if np.any(pred_mask) else []

        #  Add raw mask to the response
        raw_mask_encoded = base64.b64encode(pred_mask.astype(np.uint8)).decode("utf-8")

        return {
            "success": True,
            "detections": detections,
            "annotated_image": encode_image(overlayed),
            "mask": raw_mask_encoded  #  new field
        }

    except Exception as e:
        logger.error(f" Error during segmentation: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ---------- Utility Functions ----------
def encode_image(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return base64_image

# ---------- CLI ----------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python segmentation_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f" Image file not found: {image_path}")
        sys.exit(1)

    if not load_segmentation_model():
        print(" Failed to load model.")
        sys.exit(1)

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_image}"

    result = run_segmentation_with_masks(data_url)

    if result["success"]:
        print(" Segmentation ran successfully.")
        print("Detections:", result["detections"])
        with open("output.jpg", "wb") as f:
            f.write(base64.b64decode(result["annotated_image"]))
        print(" Saved output as output.jpg")
    else:
        print(" Error:", result.get("error"))
