import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from typing import Any
from PIL import Image, ImageTk
import os
import torch
import shutil
import numpy as np
from torchvision import transforms
from torchvision import utils as vutils
from textAwareMultiGan.definitions.gan import TextAwareMultiGan
from textAwareMultiGan.training.saver import save_imgs
from textDetection.definitions.uNetTextDetection import UNetTextDetection, load_pretrained_model

canvas_size = (256, 256)

def conta_file_in_cartella(cartella):
    try:
        elenco_file = os.listdir(cartella)
        numero_file = len(elenco_file)
        return numero_file
    except FileNotFoundError:
        return -1  # Cartella non trovata
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        return -1  # Altro errore

def init():
    pwd = os.getcwd()
    folder = os.path.join(pwd, "resrcs/neuralNetworks/textAwareMultiGan")
    #Controllo che esista la cartella contenente i pesi della gan. Se esiste carico i pesi
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
    #Controllo che esista la cartella contenente i pesi della text detection. Se esiste carico i pesi
    folder = os.path.join(pwd, "resrcs/neuralNetworks/textDetection")
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        list_files = ["text_detection.pth"]
        ret, file = check_file_in_folder(list_files, folder)
        if not ret:
            unet.load_state_dict(torch.load(os.path.join(folder, "text_detection.pth"), map_location=torch.device('cpu')))
            text_detection_setting.set(folder)
            messagebox.showinfo("Success", "UNET loaded")
                      

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
        #text_detection_setting = filedialog.askdirectory(title=title)
        file_td = filedialog.askopenfilename(title=title, filetypes=[(".pth files", "*.pth")])
        if text_detection_setting != "":
            unet.load_state_dict(torch.load(file_td, map_location=torch.device('cpu')))
            messagebox.showinfo("Success", "Text Detection loaded")
        text_detection_setting.set(file_td)

def exec_text_detection(i):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    tensor_image = transform(i)
    masked_image = np.array(transforms.Resize((256, 256))(i))
    tensor_image = (tensor_image-torch.mean(tensor_image))/torch.std(tensor_image)
    tensor_image_batched = tensor_image.unsqueeze(0)

    result = unet(tensor_image_batched)
    res = torch.squeeze(result.detach().to(device))
    res = (res>0.1).float()

    for i in range(3):
        masked_image[:, :, i] = (1 - np.array(res)) * np.array(masked_image[:, :, i])
            
    return masked_image, res

def exec(option, canvas):
    gan_folder = gan_setting.get()
    unet_folder = text_detection_setting.get()
    if image.get() == "":
        messagebox.showerror("Error", f"Load an image")
        return 1
    i = Image.open(image.get())
    if len(i.mode)!= 3:
        messagebox.showerror("Error", "Wrong number of channels for image")
        return 1
    if option == 3:
        if mask_image.get() == "":
            messagebox.showerror("Error", f"Load a mask")
            return 1
        m = Image.open(mask_image.get())
    if (option == 1 or option == 2) and unet_folder == "":
        messagebox.showerror("Error", f"Select a folder containing the model file from the button \"Setting TextDetection\"")
        return 1
    elif option == 3 and gan_folder == "":
        messagebox.showerror("Error", f"Select a folder containing the generators file from the button \"Setting Inpainting\"")
        return 1
    if option == 1:
        result, mask = exec_text_detection(i)
        im = Image.fromarray(result)
    elif option == 2:
        res, mask = exec_text_detection(i) 
        arr_pred=(np.array(mask)* 255).astype(np.uint8)
        mask = Image.fromarray(arr_pred)
        maskered = gan.pre_processing(i, mask)        
        result = gan(maskered)
        res = vutils.make_grid(result.detach(), normalize=True)
        im = res.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(im)
    elif option == 3:
        maskered = gan.pre_processing(i, m)
        result = gan(maskered)
        res = vutils.make_grid(result.detach(), normalize=True)
        im = res.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(im)

    #im.save(os.path.join("resrcs/generated", "image_" + str(conta_file_in_cartella("resrcs/generated") + 1) + ".png"))
    im = im.resize(canvas_size)
    image_tk = ImageTk.PhotoImage(im)
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
        img = img.resize(canvas_size, Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        # Mostra l'immagine
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

        if not flag:
            image.set(filepath)
        else:
            mask_image.set(filepath)

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

    global unet
    unet = UNetTextDetection()
    global gan_setting
    gan_setting = tk.StringVar()
    global text_detection_setting
    text_detection_setting = tk.StringVar()
    global gan
    gan = TextAwareMultiGan(res=256)
    global image
    image = tk.StringVar()
    global mask_image
    mask_image = tk.StringVar()

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
