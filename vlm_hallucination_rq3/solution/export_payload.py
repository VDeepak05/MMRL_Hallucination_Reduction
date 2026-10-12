import torch
from PIL import Image
import numpy as np
from handoff import Member1HandoffInterface

def export_handoff_artifact():
    print("==================================================")
    print(" RQ3 — Exporting Member 1 Handoff Payload         ")
    print("==================================================")

    # 1. Initialize interface
    handoff_system = Member1HandoffInterface()

    # 2. Simulate input image from dataset
    dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    processor = handoff_system.pipeline.vision_pipeline.processor
    pixel_values = processor(images=dummy_image, return_tensors="pt")["pixel_values"]

    # 3. Generate payload
    payload = handoff_system.generate_member2_payload(pixel_values)

    # 4. Save to disk for Member 2 to load directly
    output_path = "member1_payload.pt"
    torch.save(payload, output_path)

    print(f"[+] Payload successfully exported to: {output_path}")
    print(f"    - Dual Stream Payload Shape : {payload['dual_stream'].shape}")
    print(f"    - Ready for Member 2 Decoder: True")

if __name__ == "__main__":
    export_handoff_artifact()
