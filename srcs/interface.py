import tkinter as tk
from tkinter import filedialog
from typing import Any
from PIL import Image, ImageTk

canvas_size = (256, 256)
text_detection_setting = ""
gan_setting = ""

def open(options):
    if options.get() == 1:
        text_detection_setting = filedialog.askdirectory(title="Select a folder for text detection",)
    elif options.get() == 2:
        gan_setting = filedialog.askdirectory(title="Select a folder for text detection",)
        text_detection_setting = filedialog.askdirectory(title="Select a folder for GAN",)
    elif options.get() == 3:
        gan_setting = filedialog.askdirectory(title="Select a folder for GAN",)

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

def radio_button(frame_top):

    opzioni_var = tk.IntVar()
    opzione_text_detection = tk.Radiobutton(frame_top, text="Text Detection", variable=opzioni_var, value=1)
    opzione_text_detection.pack(side="left", padx=10)

    opzione_text_removal = tk.Radiobutton(frame_top, text="Text Removal", variable=opzioni_var, value=2)
    opzione_text_removal.pack(side="left", padx=10)

    opzione_inpainting = tk.Radiobutton(frame_top, text="Inpainting", variable=opzioni_var, value=3)
    opzione_inpainting.pack(side="left", padx=10)

    return opzioni_var

def button(frame, text, command):
    button = tk.Button(frame, text=text, command=command)
    button.pack(pady=10)

def frame_bottom(finestra):
    frame_inferiore = tk.Frame(finestra)
    frame_inferiore.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

    frame_sinistra = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    frame_sinistra.pack(side="left", fill="both", expand=True)

    canvas_left = tk.Canvas(frame_sinistra, width=canvas_size[0], height=canvas_size[1])
    canvas_left.pack()

    frame_center = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    frame_center.pack(side="left", fill="both", expand=True)

    canvas_center = tk.Canvas(frame_center, width=canvas_size[0], height=canvas_size[1])
    canvas_center.pack()

    button_load_mask = tk.Button(frame_center, text="Load Mask", command = lambda: carica_immagine(canvas_left))
    button_load_mask.pack(pady=10)

    frame_destra = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    frame_destra.pack(side="left", fill="both", expand=True)

    canvas_right = tk.Canvas(frame_destra, width=canvas_size[0], height=canvas_size[1])
    canvas_right.pack()

    pulsante_carica_immagine = tk.Button(frame_sinistra, text="Load Image", command = lambda: carica_immagine(canvas_left))
    pulsante_carica_immagine.pack(pady=10)

    return frame_center

def window_ex():
    # Crea la finestra principale
    finestra = tk.Tk()
    finestra.title("Interfaccia per Operazioni di Immagini")

    top_frame = tk.Frame(finestra)
    top_frame.pack(side="top", fill="x")

    option = radio_button(top_frame)
    button(top_frame, "Exec", command=lambda: esegui_operazione(option))
    button(top_frame, "Setting", command=lambda: open(option))
    frame_bottom(finestra)


    # Avvia l'applicazione
    finestra.mainloop()
