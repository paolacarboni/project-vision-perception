import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from typing import Any
from PIL import Image, ImageTk
import os
import torch
import shutil
from torchvision import transforms
from textAwareMultiGan.definitions.gan256 import GAN256
from textAwareMultiGan.training.saver import save_imgs

canvas_size = (256, 256)

def init():
    pwd = os.getcwd()
    folder = os.path.join(pwd, "resrcs/neuralNetworks/textAwareMultiGan")
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        list_files = [ "generator_32.pth", "generator_64.pth", "generator_128.pth", "generator_256.pth"]
        ret, file = check_file_in_folder(list_files, folder)
        if not ret:
            for i in range(4):
                gan.load_G_weights(os.path.join(folder, list_files[i]), i)
            gan_setting.set(folder)
            messagebox.showinfo("Success", "GAN loaded")

def check_file_in_folder(files, folder):
    for file in files:
        if file not in os.listdir(folder):
            return 1, file
    return 0, ""

def open(flag, title):
    if flag:
        folder = filedialog.askdirectory(title=title)
        list_files = [ "generator_32.pth", "generator_64.pth", "generator_128.pth", "generator_256.pth"]
        ret, file = check_file_in_folder(list_files, folder)
        if ret:
            messagebox.showerror("Error", f"The folder does not contain the file '{file}'")
            gan_setting.set("")
        else:
            for i in range(4):
                file = os.path.join(folder, list_files[i])
                gan.load_G_weights(file, i)
                shutil.copy(file, os.path.join(os.getcwd(), "resrcs/neuralNetworks/textAwareMultiGan"))
            gan_setting.set(folder)
            messagebox.showinfo("Success", "GAN loaded")
    else:
        text_detection_setting = filedialog.askdirectory(title=title)

def exec(option, canvas):
    print("GAN SETTING:", gan_setting.get())
    folder = gan_setting.get()
    if option == 1:
        pass
    elif option == 2:
        pass
    elif option == 3:
        if folder == "":
            messagebox.showerror("Error", f"Select a folder containing the generators file from the button \"Setting Inpainting\"")
        elif image.get() == "":
            messagebox.showerror("Error", f"Load an image")
        elif mask.get() == "":
            messagebox.showerror("Error", f"Load a mask")
        else:
            i = Image.open(image.get())
            m = Image.open(mask.get())
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

            transformM = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Grayscale(num_output_channels=1),
            ])

            tensor_image = transform(i)
            tensor_mask = transformM(m)
            tensor_image_batched = tensor_image.unsqueeze(0)
            tensor_mask_batched = tensor_mask.unsqueeze(0)
            result = gan.forward(tensor_image_batched, tensor_mask_batched)
            save_imgs(result, "img")
            print(result[0])
            valore_massimo = result[0].max()
            valore_minimo = result[0].min()

            # Normalizza il tensore nell'intervallo [0, 1]
            tensor_normalizzato = (result[0] - valore_minimo) / (valore_massimo - valore_minimo)
            #blended_image = tensor_image * tensor_mask + result[0] * (1 - tensor_mask)
            img = Image.fromarray(255 * result[0].permute(1, 2, 0).detach().numpy().astype('uint8'))
            image_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            canvas.image = image_tk


def esegui_operazione():
    selection = opzioni_var.get()
    if selection == 1:
        frame_center.pack_forget()
    elif selection == 2:
        frame_center.pack_forget()
    elif selection == 3:
        frame_destra.pack_forget()
        frame_center.pack(side="left", fill="both", expand=True)
        frame_destra.pack(side="left", fill="both", expand=True)

def carica_immagine(flag, canvas):
    filepath = filedialog.askopenfilename(filetypes=[("Immagini", "*.png;*.jpg;*.jpeg;*.bmp")])
    if filepath:
        img = Image.open(filepath)
        img = img.resize(canvas_size, Image.ANTIALIAS)  # Ridimensiona l'immagine se necessario
        img = ImageTk.PhotoImage(img)

        # Mostra l'immagine
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

        if not flag:
            image.set(filepath)
        else:
            mask.set(filepath)

def radio_button(frame_top):

    opzioni_var.set(1)

    opzione_text_detection = tk.Radiobutton(frame_top, text="Text Detection", variable=opzioni_var, value=1, command=esegui_operazione)
    opzione_text_detection.pack(side="left", padx=10)

    opzione_text_removal = tk.Radiobutton(frame_top, text="Text Removal", variable=opzioni_var, value=2, command=esegui_operazione)
    opzione_text_removal.pack(side="left", padx=10)

    opzione_inpainting = tk.Radiobutton(frame_top, text="Inpainting", variable=opzioni_var, value=3, command=esegui_operazione)
    opzione_inpainting.pack(side="left", padx=10)

    return opzioni_var

def button(frame, text, command):
    button = tk.Button(frame, text=text, command=command)
    button.pack(pady=10)
    return button

def buttons(frame, option):

    s_t_b = button(frame, "Setting TextDetection", command=lambda: open(0, "Select directory for text detection"))
    s_i_b = button(frame, "Setting Inpainting", command=lambda: open(1, "Select directory for inpainting"))

    s_t_b.pack(side=tk.LEFT, padx=10)
    s_i_b.pack(side=tk.LEFT, padx=10)

def frame_bottom(finestra):
    frame_inferiore = tk.Frame(finestra)
    frame_inferiore.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

    frame_sinistra = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    frame_sinistra.pack(side="left", fill="both", expand=True)

    canvas_left = tk.Canvas(frame_sinistra, width=canvas_size[0], height=canvas_size[1])
    canvas_left.pack()

    global frame_center
    frame_center = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    #frame_center.pack(side="left", fill="both", expand=True)

    canvas_center = tk.Canvas(frame_center, width=canvas_size[0], height=canvas_size[1])
    canvas_center.pack()

    button_load_mask = tk.Button(frame_center, text="Load Mask", command = lambda: carica_immagine(1, canvas_center))
    button_load_mask.pack(pady=10)

    global frame_destra
    frame_destra = tk.Frame(frame_inferiore, width=50, height=50, relief="sunken", borderwidth=1)
    frame_destra.pack(side="left", fill="both", expand=True)

    canvas_right = tk.Canvas(frame_destra, width=canvas_size[0], height=canvas_size[1])
    canvas_right.pack()

    pulsante_carica_immagine = tk.Button(frame_sinistra, text="Load Image", command = lambda: carica_immagine(0, canvas_left))
    pulsante_carica_immagine.pack(pady=10)

    e_b = button(frame_destra, "Exec", command=lambda: exec(opzioni_var.get(), canvas_right))
    e_b.pack(pady=10)

    return frame_center

def window_ex():

    # Crea la finestra principale
    finestra = tk.Tk()
    finestra.title("Interfaccia per Operazioni di Immagini")

    global gan_setting
    gan_setting = tk.StringVar()
    global text_detection_setting
    text_detection_setting = tk.StringVar()
    global gan
    gan = GAN256(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    global image
    image = tk.StringVar()
    global mask
    mask = tk.StringVar()

    global opzioni_var
    opzioni_var = tk.IntVar()

    top_frame = tk.Frame(finestra)
    top_frame.pack(side="top", fill="x")

    option = radio_button(top_frame)
    buttons(top_frame, option)
    frame_bottom(finestra)

    init()

    # Avvia l'applicazione
    finestra.mainloop()
