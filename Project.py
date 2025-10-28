import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image
import cv2
import pytesseract
import numpy as np
from sklearn.decomposition import PCA
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# ---------------- OCR FUNCTIONS ----------------
def recognize_letter(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip() if text.strip() else "Could not recognize letter"
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_word(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        config = '--psm 8'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip() if text.strip() else "Could not recognize word"
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_text(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        config = '--psm 6'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip() if text.strip() else "Could not recognize text"
    except Exception as e:
        return f"Error: {str(e)}"

def apply_pca_enhancement(image_path, n_components=50):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        h, w = thresh.shape
        data = np.array([thresh[i, :] for i in range(h)])
        pca = PCA(n_components=min(n_components, min(h, w)))
        pca_result = pca.fit_transform(data)
        reconstructed = pca.inverse_transform(pca_result).reshape(h, w)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return reconstructed, pca.explained_variance_ratio_
    except Exception as e:
        print(f"PCA Error: {str(e)}")
        return None, None

def recognize_with_pca(image_path, recognition_type='text'):
    try:
        enhanced_img, variance_ratio = apply_pca_enhancement(image_path)
        if enhanced_img is None:
            return "PCA enhancement failed"
        pil_img = Image.fromarray(enhanced_img)
        if recognition_type == 'letter':
            config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        elif recognition_type == 'word':
            config = '--psm 8'
        else:
            config = '--psm 6'
        text = pytesseract.image_to_string(pil_img, config=config)
        result = text.strip()
        return result if result else f"Could not recognize {recognition_type}"
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_accuracy(predicted, actual):
    if not predicted or not actual:
        return 0.0
    return SequenceMatcher(None, predicted.lower(), actual.lower()).ratio() * 100

# ---------------- GUI ----------------
class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TextVision - OCR + PCA Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2f")
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        ttk.Label(self.root, text="TextVision", font=("Helvetica", 22, "bold"), foreground="white", background="#1e1e2f").pack(pady=10)
        
        # Frame for file selection
        frame1 = tk.Frame(self.root, bg="#2e2e3f")
        frame1.pack(pady=10, fill="x", padx=20)
        
        self.file_label = ttk.Label(frame1, text="No file selected", background="#2e2e3f", foreground="white")
        self.file_label.pack(side="left", padx=10)
        
        ttk.Button(frame1, text="Select Image", command=self.select_file).pack(side="right", padx=10)
        
        # Ground truth input
        frame2 = tk.Frame(self.root, bg="#2e2e3f")
        frame2.pack(pady=10, fill="x", padx=20)
        
        ttk.Label(frame2, text="Expected Text:", background="#2e2e3f", foreground="white").pack(side="left", padx=5)
        self.gt_entry = ttk.Entry(frame2, width=50)
        self.gt_entry.pack(side="left", padx=5)
        
        ttk.Button(frame2, text="Run OCR Analysis", command=self.run_analysis).pack(side="right", padx=10)
        ttk.Button(frame2, text="Show Accuracy Graphs", command=self.open_graph_window).pack(side="right", padx=10)
        
        # Results area
        self.results_frame = tk.Frame(self.root, bg="#1e1e2f")
        self.results_frame.pack(pady=20, fill="both", expand=True)
        
        self.results_text = tk.Text(self.results_frame, bg="#2e2e3f", fg="white", font=("Consolas", 12))
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if file_path:
            self.image_path = file_path
            self.file_label.config(text=file_path)
    
    def run_analysis(self):
        self.results_text.delete("1.0", tk.END)
        if not hasattr(self, "image_path"):
            messagebox.showwarning("Warning", "Please select an image file first!")
            return
        ground_truth = self.gt_entry.get()
        if not ground_truth:
            messagebox.showwarning("Warning", "Please enter the expected text!")
            return
        
        self.results_text.insert(tk.END, "Running OCR Analysis...\n\n")
        self.root.update()
        
        start_time = time.time()
        original_text = recognize_text(self.image_path)
        pca_text = recognize_with_pca(self.image_path, "text")
        end_time = time.time()
        accuracy_original = calculate_accuracy(original_text, ground_truth)
        accuracy_pca = calculate_accuracy(pca_text, ground_truth)
        
        self.results_text.insert(tk.END, f"Original OCR Text: {original_text}\n")
        self.results_text.insert(tk.END, f"PCA-Enhanced OCR Text: {pca_text}\n")
        self.results_text.insert(tk.END, f"\nAccuracy Original: {accuracy_original:.2f}%\n")
        self.results_text.insert(tk.END, f"Accuracy PCA: {accuracy_pca:.2f}%\n")
        self.results_text.insert(tk.END, f"\nProcessing Time: {end_time - start_time:.3f} seconds\n\n")
    
    def open_graph_window(self):
        if not hasattr(self, "image_path"):
            messagebox.showwarning("Warning", "Please select an image file first!")
            return
        ground_truth = self.gt_entry.get()
        if not ground_truth:
            messagebox.showwarning("Warning", "Please enter the expected text!")
            return
        
        graph_win = tk.Toplevel(self.root)
        graph_win.title("OCR Accuracy Dashboard")
        graph_win.geometry("1300x900")
        graph_win.configure(bg="#1e1e2f")
        
        notebook = ttk.Notebook(graph_win)
        notebook.pack(fill="both", expand=True)
        
        # Tabs for 4 graphs
        tabs = ["Original Methods Accuracy", "Processing Time", "PCA Components vs Accuracy", "Top 5 Methods"]
        for tab_name in tabs:
            frame = tk.Frame(notebook, bg="#1e1e2f")
            notebook.add(frame, text=tab_name)
        
        # --- 1. Original Methods Accuracy ---
        methods = ["Letter", "Word", "Text"]
        accuracies = [calculate_accuracy(recognize_letter(self.image_path), ground_truth),
                      calculate_accuracy(recognize_word(self.image_path), ground_truth),
                      calculate_accuracy(recognize_text(self.image_path), ground_truth)]
        fig1 = plt.Figure(figsize=(7,5))
        ax1 = fig1.add_subplot(111)
        ax1.bar(methods, accuracies, color="#4ECDC4")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 100)
        canvas1 = FigureCanvasTkAgg(fig1, master=notebook.nametowidget(notebook.tabs()[0]))
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        
        # --- 2. Processing Time ---
        times = []
        for func in [recognize_letter, recognize_word, recognize_text]:
            start = time.time()
            func(self.image_path)
            times.append(time.time() - start)
        fig2 = plt.Figure(figsize=(7,5))
        ax2 = fig2.add_subplot(111)
        ax2.bar(methods, times, color="#F39C12")
        ax2.set_ylabel("Time (s)")
        canvas2 = FigureCanvasTkAgg(fig2, master=notebook.nametowidget(notebook.tabs()[1]))
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        
        # --- 3. PCA Components vs Accuracy ---
        pca_components = [10,20,30,50,75,100]
        pca_accuracies = []
        for n in pca_components:
            enhanced_img,_ = apply_pca_enhancement(self.image_path,n)
            if enhanced_img is not None:
                text = pytesseract.image_to_string(Image.fromarray(enhanced_img), config='--psm 6').strip()
                acc = calculate_accuracy(text, ground_truth)
            else:
                acc = 0
            pca_accuracies.append(acc)
        fig3 = plt.Figure(figsize=(7,5))
        ax3 = fig3.add_subplot(111)
        ax3.plot(pca_components,pca_accuracies,marker='o',color="#E74C3C")
        ax3.set_xlabel("Number of PCA Components")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_ylim(0,100)
        canvas3 = FigureCanvasTkAgg(fig3, master=notebook.nametowidget(notebook.tabs()[2]))
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill="both", expand=True)
        
        # --- 4. Top 5 Performing Methods ---
        all_methods = methods + [f"PCA-{c}" for c in pca_components]
        all_accuracies = accuracies + pca_accuracies
        top_indices = np.argsort(all_accuracies)[-5:]
        top_methods = [all_methods[i] for i in top_indices]
        top_acc = [all_accuracies[i] for i in top_indices]
        fig4 = plt.Figure(figsize=(7,5))
        ax4 = fig4.add_subplot(111)
        ax4.barh(top_methods, top_acc, color="#27AE60")
        ax4.set_xlabel("Accuracy (%)")
        ax4.set_xlim(0,100)
        canvas4 = FigureCanvasTkAgg(fig4, master=notebook.nametowidget(notebook.tabs()[3]))
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill="both", expand=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use('clam')
    app = OCRApp(root)
    root.mainloop()
