import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.gridspec import GridSpec

# Percorso base
base_path = r"C:\Users\aless\Desktop\TD_ale\TD\Es_11\presentazione"

# Lista dei file originali
file_names = [
    "task4_oscillazione_attorno_minimo_ampl_0_1_presentation.png",
    "task4_limite_superamento_barriera_ampl_0_33_presentation.png",
    "task4_oscillazioni_attorno_minimi_ampl_0_34_presentation.png",
    "task4_allineamento_cuspidi_ampl_0_66_presentation.png",
    "task4_smussamento_finale_cuspidi_ampl_2_presentation.png",
    "task4_saturazione_opamp_ampl_4_presentation.png"
]

# Etichette delle ampiezze
labels = ["0.1 V", "0.33 V", "0.34 V", "0.66 V", "2.0 V", "4.0 V"]
files = [os.path.join(base_path, f) for f in file_names]

# Aumentiamo ancora la dimensione della figura per dare più respiro
fig = plt.figure(figsize=(28, 15))

# Ottimizzazione aggressiva degli spazi con GridSpec:
# ridotti margini esterni (left, right, bottom) e aumentato top per il titolo principale
# ridotti hspace e wspace per avvicinare i grafici tra loro
gs = GridSpec(2, 3, figure=fig, left=0.02, right=0.98, bottom=0.02, top=0.90, hspace=0.2, wspace=0.08)

# Titolo principale più grande e ben posizionato
fig.suptitle("Traiettoria nello spazio delle fasi", fontsize=42, fontweight='bold', y=0.97)

for i, path in enumerate(files):
    # Assegna il subplot alla cella corrente della griglia
    ax = fig.add_subplot(gs[i])
    
    if os.path.exists(path):
        # Carica e visualizza l'immagine con interpolazione Lanczos per la nitidezza
        img = mpimg.imread(path)
        ax.imshow(img, interpolation='lanczos')
        
        # Titolo dell'ampiezza: fontsize aumentato e pad ridotto per avvicinarlo al grafico
        ax.set_title(f"Ampiezza: {labels[i]}", fontsize=28, fontweight='bold', pad=5)
        
        # Nascondi gli assi di matplotlib
        ax.axis('off')
    else:
        # Messaggio di errore se il file non è trovato
        ax.text(0.5, 0.5, f"File non trovato:\n{os.path.basename(path)}", 
                ha='center', va='center', color='red', fontsize=18)
        ax.axis('off')

# Salvataggio in PDF ad alta risoluzione (300 dpi)
output_pdf = os.path.join(base_path, "collage_fasi_massimizzato.pdf")
plt.savefig(output_pdf, dpi=300)

print(f"Collage massimizzato salvato con successo in: {output_pdf}")
# Visualizza la figura
plt.show()