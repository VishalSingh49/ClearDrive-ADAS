import cv2

def enhance_low_light(img):
    """Raat ke time ya andhere mein image ko saaf karna."""
    # Image ko LAB color space mein convert karo (L = Lightness)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE apply karo sirf L channel par
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Wapas merge karke BGR mein convert karo
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced