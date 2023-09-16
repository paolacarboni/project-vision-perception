import tkinter as tk
from tkinter import filedialog
from typing import Any
from PIL import Image, ImageTk

canvas_size = (300, 300)

def esegui_operazione(opzioni_var):
    selezione = opzioni_var.get()
    if selezione == 1:
        # Esegui l'operazione di Text Detection
        print("Hai selezionato Text Detection")
    elif selezione == 2:
        # Esegui l'operazione di Text Removal
        print("Hai selezionato Text Removal")
    elif selezione == 3:
        # Esegui l'operazione di Inpainting
        print("Hai selezionato Inpainting")

def carica_immagine(canvas):
    filepath = filedialog.askopenfilename(filetypes=[("Immagini", "*.png;*.jpg;*.jpeg;*.bmp")])
    if filepath:
        img = Image.open(filepath)
        img = img.resize(canvas_size, Image.ANTIALIAS)  # Ridimensiona l'immagine se necessario
        img = ImageTk.PhotoImage(img)

        # Mostra l'immagine
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

def frame_top(finestra):
    frame_superiore = tk.Frame(finestra)
    frame_superiore.pack(side="top", fill="x")

    opzioni_var = tk.IntVar()
    opzione_text_detection = tk.Radiobutton(frame_superiore, text="Text Detection", variable=opzioni_var, value=1)
    opzione_text_detection.pack(side="left", padx=10)

    opzione_text_removal = tk.Radiobutton(frame_superiore, text="Text Removal", variable=opzioni_var, value=2)
    opzione_text_removal.pack(side="left", padx=10)

    opzione_inpainting = tk.Radiobutton(frame_superiore, text="Inpainting", variable=opzioni_var, value=3)
    opzione_inpainting.pack(side="left", padx=10)

    pulsante_esegui = tk.Button(frame_superiore, text="Esegui Operazione", command=lambda: esegui_operazione(opzioni_var))
    pulsante_esegui.pack(pady=10)

def frame_bottom(finestra):
    frame_inferiore = tk.Frame(finestra)
    frame_inferiore.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

    frame_sinistra = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    frame_sinistra.pack(side="left", fill="both", expand=True)

    canvas_left = tk.Canvas(frame_sinistra, width=canvas_size[0], height=canvas_size[1])
    canvas_left.pack()

    frame_destra = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    frame_destra.pack(side="left", fill="both", expand=True)

    canvas_right = tk.Canvas(frame_destra, width=300, height=300)
    canvas_right.pack()

    pulsante_carica_immagine = tk.Button(frame_sinistra, text="Carica Immagine", command = lambda: carica_immagine(canvas_left))
    pulsante_carica_immagine.pack(pady=10)

def window_ex():
    # Crea la finestra principale
    finestra = tk.Tk()
    finestra.title("Interfaccia per Operazioni di Immagini")

    frame_top(finestra)
    frame_bottom(finestra)

    # Avvia l'applicazione
    finestra.mainloop()
