from PIL import Image

path = '/home/marco/Desktop/Uni_anno3/TD/Es_09/logbook/'

img = Image.open(path+"fterzi_divider_task4_es09.png")
img = img.convert("RGB")
img.save(path+"Divisore_fterzi.pdf")