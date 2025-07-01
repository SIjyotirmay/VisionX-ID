import cv2, os, numpy as np

ROOT_DIR   = "dataset"          # your cropped‑face dataset root
ROT_ANGLE  = 15                 # degrees for the “angle” variant
BLUR_KSIZE = (5, 5)             # Gaussian blur kernel

def simulate_night(img):
    g   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g   = cv2.equalizeHist(g)
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    out = cv2.convertScaleAbs(out, alpha=0.4, beta=0)  
    return out

def rotate(img, deg=ROT_ANGLE):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

def blur(img):
    return cv2.GaussianBlur(img, BLUR_KSIZE, 0)

def already_done(base, folder):
    return os.path.exists(os.path.join(folder, f"{base}_night.jpg"))

for person in os.listdir(ROOT_DIR):
    pdir = os.path.join(ROOT_DIR, person)
    if not os.path.isdir(pdir): continue

    print(f"▶️  {person}")
    for fname in os.listdir(pdir):
        if not fname.lower().endswith(('.jpg', '.png')): continue
        base, ext = os.path.splitext(fname)
        if "_night" in base or "_angle" in base or "_flip" in base:
            continue  # skip previously‑augmented files
        if already_done(base, pdir):
            continue  # avoid duplicates

        img_path = os.path.join(pdir, fname)
        img      = cv2.imread(img_path)
        if img is None: 
            print("  ⚠️  unreadable:", fname); continue

        # 2. night
        night = simulate_night(img)
        cv2.imwrite(os.path.join(pdir, f"{base}_night.jpg"), night)

        # 3. angle (day)
        ang  = rotate(img)
        cv2.imwrite(os.path.join(pdir, f"{base}_angle.jpg"), ang)

        # 4. angle + night
        ang_n = simulate_night(ang)
        cv2.imwrite(os.path.join(pdir, f"{base}_angle_night.jpg"), ang_n)

        # 5. flip (day)
        flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(pdir, f"{base}_flip.jpg"), flip)

        # 6. flip + night
        flip_n = simulate_night(flip)
        cv2.imwrite(os.path.join(pdir, f"{base}_flip_night.jpg"), flip_n)

        # 7. night + blur
        night_blur = blur(night)
        cv2.imwrite(os.path.join(pdir, f"{base}_night_blur.jpg"), night_blur)

print("✅  Augmentation complete – new variants saved beside originals.")
