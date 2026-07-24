import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.transforms import ToTensor
import torch.nn as nn
import pathlib
import json

PROJECT_ROOT = pathlib.Path(__file__).parent

IMG_HEIGHT, IMG_WIDTH = 64, 200

def load_vocab():
    """Load vocabulary from JSON file created by train_ocr.py"""
    vocab_path = PROJECT_ROOT / "models" / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found at {vocab_path}\n"
            "Please run train_ocr.py first to generate it."
        )
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
    char_to_idx = vocab_data['char_to_idx']
    vocab_size = vocab_data['vocab_size']
    
    return char_to_idx, idx_to_char, vocab_size


class CRNN(nn.Module):
    """Exact architecture from train_ocr.py"""
    def __init__(self, vocab_size):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv1.stride = (1, 1)
        if resnet.layer4[0].downsample is not None:
            resnet.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        with torch.no_grad():
            dummy = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH)
            out = self.backbone(dummy)
            out_channels = out.size(1)
            out_height = out.size(2)
            self.out_width = out.size(3)

        self.lstm = nn.LSTM(
            input_size=out_channels * out_height,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.backbone(x)
        batch, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, w, c * h)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.permute(1, 0, 2).log_softmax(2)


def decode_predictions(log_probs, idx_to_char):
    preds = log_probs.argmax(2).permute(1, 0).cpu().numpy()
    texts = []
    for pred in preds:
        text = []
        prev_char = None
        for idx in pred:
            if idx == 0:
                prev_char = None
                continue
            if idx != prev_char:
                text.append(idx_to_char.get(idx, ''))
                prev_char = idx
        texts.append(''.join(text))
    return texts


class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection & OCR")
        self.root.geometry("1000x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        self.ocr_model = None
        self.idx_to_char = None
        
        self.load_models()

        self.canvas = tk.Canvas(root, width=800, height=500, bg="white")
        self.canvas.pack(pady=10)

        self.load_btn = tk.Button(root, text="Load Image", command=self.load_image, width=20, height=2)
        self.load_btn.pack(pady=5)

        self.result_label = tk.Label(root, text="License Plates:", font=("Arial", 12))
        self.result_label.pack(pady=5)

        self.result_text = tk.Text(root, height=10, width=80, state="disabled")
        self.result_text.pack(pady=5)

        self.car_count_label = tk.Label(root, text="Total Cars Detected: 0", font=("Arial", 12, "bold"))
        self.car_count_label.pack(pady=5)

        self.current_image = None
        self.current_image_tk = None

    def load_models(self):
        detector_path = PROJECT_ROOT / "models" / "best_detector.pth"
        if detector_path.exists():
            try:
                self.detector = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
                in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
                self.detector.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
                
                checkpoint = torch.load(detector_path, map_location=self.device, weights_only=False)
                self.detector.load_state_dict(checkpoint['model_state_dict'])
                self.detector.to(self.device)
                self.detector.eval()
                print("Detector loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load detector: {e}")
        else:
            messagebox.showwarning("Warning", f"Detector not found at {detector_path}")

        ocr_path = PROJECT_ROOT / "models" / "best_ocr.pth"
        if ocr_path.exists():
            try:
                char_to_idx, self.idx_to_char, vocab_size = load_vocab()
                
                self.ocr_model = CRNN(vocab_size)
                
                checkpoint = torch.load(ocr_path, map_location=self.device, weights_only=False)
                self.ocr_model.load_state_dict(checkpoint['model_state_dict'])
                self.ocr_model.to(self.device)
                self.ocr_model.eval()
                print(f"OCR model loaded successfully (vocab size: {vocab_size}, sequence length: {self.ocr_model.out_width})")
            except FileNotFoundError as e:
                messagebox.showerror("Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load OCR model: {e}")
                import traceback
                traceback.print_exc()
        else:
            messagebox.showwarning("Warning", f"OCR model not found at {ocr_path}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        try:
            self.current_image = Image.open(file_path).convert("RGB")
            img_width, img_height = self.current_image.size
            canvas_width, canvas_height = 800, 500
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            self.current_image_resized = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_image_tk = ImageTk.PhotoImage(self.current_image_resized)
            
            self.canvas.delete("all")
            self.canvas.create_image(
                (canvas_width - new_width) // 2,
                (canvas_height - new_height) // 2,
                anchor="nw",
                image=self.current_image_tk
            )
            self.process_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def process_image(self):
        if not self.detector or not self.ocr_model:
            self.result_text.config(state="normal")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Models not loaded. Please train them first.")
            self.result_text.config(state="disabled")
            return

        transform = ToTensor()
        img_tensor = transform(self.current_image).to(self.device)
        
        with torch.no_grad():
            predictions = self.detector([img_tensor])[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        
        indices = scores > 0.5
        boxes = boxes[indices]

        draw_image = self.current_image_resized.copy()
        draw = ImageDraw.Draw(draw_image)
        img_width, img_height = self.current_image.size
        canvas_width, canvas_height = 800, 500
        ratio = min(canvas_width / img_width, canvas_height / img_height)

        license_plates = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            
            draw.rectangle(
                [(x_min * ratio, y_min * ratio), (x_max * ratio, y_max * ratio)], 
                outline="red", 
                width=2
            )
            
            crop_img = self.current_image.crop((x_min, y_min, x_max, y_max))
            ocr_text = self.recognize_plate(crop_img)
            license_plates.append(ocr_text)

        self.current_image_tk = ImageTk.PhotoImage(draw_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            (canvas_width - draw_image.width) // 2,
            (canvas_height - draw_image.height) // 2,
            anchor="nw",
            image=self.current_image_tk
        )

        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        for i, plate in enumerate(license_plates):
            self.result_text.insert(tk.END, f"Plate {i+1}: {plate}\n")
        self.result_text.config(state="disabled")
        
        self.car_count_label.config(text=f"Total Cars Detected: {len(license_plates)}")

    def recognize_plate(self, img):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.ocr_model(img_tensor)
            
        texts = decode_predictions(log_probs, self.idx_to_char)
        return texts[0] if texts else ""


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()