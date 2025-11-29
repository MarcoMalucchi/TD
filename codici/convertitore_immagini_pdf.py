from PIL import Image

path = '/home/marco/Desktop/Uni_anno3/TD/Es_08/logbook/'

img = Image.open(path+"XOR_ingresso_uscita.png")
img = img.convert("RGB")
img.save(path+"XOR_ingresso_uscita.pdf")