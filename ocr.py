import pytesseract
from PIL import Image
import os




def ocr_image(path, lang='eng'):
      img = Image.open(path)
      text = pytesseract.image_to_string(img, lang=lang)
      return text




def ocr_pages(image_paths, out_txt_dir='ocr_text'):
      os.makedirs(out_txt_dir, exist_ok=True)
      texts = []
      for p in image_paths:
            txt = ocr_image(p)
            base = os.path.splitext(os.path.basename(p))[0]
            outp = os.path.join(out_txt_dir, f'{base}.txt')
            with open(outp, 'w', encoding='utf-8') as f:
                  f.write(txt)
            texts.append((p, txt))
      return texts