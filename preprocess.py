import os
from pdf2image import convert_from_path
from PIL import Image, ImageOps
import numpy as np
import cv2




def pdf_to_images(pdf_path, dpi=300, output_dir='processed_images'):
      os.makedirs(output_dir, exist_ok=True)
      images = convert_from_path(pdf_path, dpi=dpi)
      out_paths = []
      for i, img in enumerate(images):
            path = os.path.join(output_dir, f'page_{i+1:03d}.png')
            img.save(path)
            out_paths.append(path)
      return out_paths




def deskew_image(path, out_path=None):
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
      coords = cv2.findNonZero(255 - thresh)
      if coords is None:
      # blank page
            if out_path:
                  Image.fromarray(img).save(out_path)
            return out_path or path
      angle = cv2.minAreaRect(coords)[-1]
      if angle < -45:
            angle = -(90 + angle)
      else:
            angle = -angle
      (h, w) = img.shape[:2]
      center = (w // 2, h // 2)
      M = cv2.getRotationMatrix2D(center, angle, 1.0)
      rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
      if out_path:
            cv2.imwrite(out_path, rotated)
      return out_path or path




def is_blank_image(path, pixel_thresh=0.98, ocr_text_length=10):
# visual check
      img = Image.open(path).convert('L')
      arr = np.array(img) / 255.0
      white_ratio = (arr > 0.98).mean()
      if white_ratio > pixel_thresh:
            return True
      # otherwise, rely on OCR length check from caller
      return False