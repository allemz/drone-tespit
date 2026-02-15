from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

# modeller
tespit_model = YOLO('models/tespit.pt')  
siniflandirma_model = YOLO('models/siniflandirma.pt')

DOSYA = 'samples/test10.jpg'

# Renkler ve Türkçe isimler
renkler = {'Aeroplanes': (255, 0, 0), 'Birds': (0, 255, 0), 'Drones': (0, 0, 255)}
turkce = {'Aeroplanes': 'Ucak', 'Birds': 'Kus', 'Drones': 'Drone'}

# Sonuçlar klasörü oluştur
os.makedirs('sonuclar', exist_ok=True)

# Resmi aç
image = Image.open(DOSYA).convert('RGB')
draw = ImageDraw.Draw(image)

# 1. tespit modeliyle objeyi bul
tespit_sonuc = tespit_model(image)[0]

print(f"Toplam {len(tespit_sonuc.boxes)} nesne bulundu\n")

sayac = {'Ucak': 0, 'Kus': 0, 'Drone': 0}

for i, box in enumerate(tespit_sonuc.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # %50 büyüt
    w, h = x2 - x1, y2 - y1
    x1_new = max(0, x1 - int(w * 0.5))
    y1_new = max(0, y1 - int(h * 0.5))
    x2_new = min(image.width, x2 + int(w * 0.5))
    y2_new = min(image.height, y2 + int(h * 0.5))
    
    # Kesit al
    kesit = image.crop((x1_new, y1_new, x2_new, y2_new))
    kesit = kesit.resize((224, 224))
    
    # 2. Sınıflandır
    sonuc = siniflandirma_model(kesit)[0]
    
    sinif_id = sonuc.probs.top1
    sinif = siniflandirma_model.names[sinif_id]
    guven = float(sonuc.probs.top1conf)
    
    # Türkçe isim
    sinif_tr = turkce[sinif]
    sayac[sinif_tr] += 1
    
    print(f"Nesne {i+1}: {sinif_tr} - %{guven*100:.1f}")
    
    # Çiz
    renk = renkler.get(sinif, (255, 255, 255))
    
    # Kutucuk kalınlığı (resim boyutuna göre)
    kalinlik = max(2, int(image.width / 400))
    draw.rectangle([x1, y1, x2, y2], outline=renk, width=kalinlik)
    
    # Yazı boyutu (kutucuk boyutuna göre)
    yazi_boyutu = max(12, int((x2-x1) / 5))
    try:
        font = ImageFont.truetype("arial.ttf", yazi_boyutu)
    except:
        font = ImageFont.load_default()
    
    # Yazı
    yazi = f"{sinif_tr} %{guven*100:.0f}"
    
    # Yazı arka planı (daha okunabilir)
    bbox = draw.textbbox((x1, y1-yazi_boyutu-10), yazi, font=font)
    draw.rectangle(bbox, fill=renk)
    draw.text((x1, y1-yazi_boyutu-10), yazi, fill=(255, 255, 255), font=font)

# Zaman damgası ile kaydet
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dosya_adi = f"sonuclar/sonuc_{timestamp}.jpg"
image.save(dosya_adi)

# Özet yazdır
print(f"\n{'='*40}")
print("ÖZET:")
print(f"{'='*40}")
for sinif, adet in sayac.items():
    if adet > 0:
        print(f"{sinif}: {adet} adet")
print(f"{'='*40}")
print(f"Sonuç kaydedildi: {dosya_adi}")