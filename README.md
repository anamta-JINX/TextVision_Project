# TextVision_Project

**TextVision** is an advanced Python-based application that combines **Optical Character Recognition (OCR)** with **Principal Component Analysis (PCA)** for text enhancement, analysis, and benchmarking. The project provides both a **GUI interface** and a **programmatic interface** to perform OCR on images, improve text recognition using PCA, and visualize performance metrics through interactive graphs.

---

## Features

1. **OCR Recognition**

   * Recognizes **letters**, **words**, or **full text** from images.
   * Uses **Tesseract OCR** via Python.
   * Supports multiple input types: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`.

2. **PCA-Enhanced OCR**

   * Applies **Principal Component Analysis (PCA)** to reduce noise and improve recognition accuracy.
   * Users can adjust the number of PCA components for fine-tuning.
   * Provides both textual output and explained variance of PCA.

3. **Accuracy Benchmarking**

   * Compares OCR accuracy of original methods versus PCA-enhanced methods.
   * Computes **similarity percentage** between recognized text and ground truth.
   * Measures **processing time** for each method.

4. **Interactive GUI**

   * Built using **Tkinter**.
   * Allows users to **select images**, **enter expected text**, and **run analysis** with a single click.
   * Shows results in a **clean, color-coded interface**.
   * Opens a **graphical dashboard** displaying:

     * Original OCR accuracy for letters, words, and text
     * Processing time comparison
     * PCA components vs accuracy plot
     * Top 5 performing methods

5. **Graph Visualization**

   * Uses **Matplotlib** integrated with Tkinter via `FigureCanvasTkAgg`.
   * Supports **tabbed interface** to view multiple graphs in one window.
   * Provides **bar graphs, line plots, and horizontal bar charts** with clear labels.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/anamta-JINX/TextVision_Project.git
cd TextVision_Project
```

2. **Install dependencies**
   Make sure you have **Python 3.x** installed.

```bash
pip install opencv-python pillow pytesseract numpy scikit-learn matplotlib
```

3. **Install Tesseract OCR**

* **Windows**: Download from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and add the installation path to your system environment variables.
* **Linux**:

```bash
sudo apt install tesseract-ocr
```

* **MacOS**:

```bash
brew install tesseract
```

---

## Usage

### 1. GUI Interface

1. Run the GUI:

```bash
python Project.py
```

2. **Select an image**: Click `Select Image` to choose your input file.
3. **Enter expected text**: Type the ground truth in the "Expected Text" field.
4. **Run OCR Analysis**: Click the button to process the image.
5. **View results**:

   * Text recognized by original OCR
   * Text recognized after PCA enhancement
   * Accuracy percentages and processing time
6. **Show Accuracy Graphs**: Opens a new window with 4 interactive tabs displaying detailed metrics.

---

### 2. Programmatic Interface

You can run a complete OCR analysis and generate graphs via Python:

```python
from Project import run_complete_analysis

image_path = "text/para2.png"
ground_truth = "Hello World"

results = run_complete_analysis(image_path, ground_truth)
```

**Output includes:**

* Original and PCA-enhanced OCR text
* Accuracy and processing times
* Saved graphs (`.png`) for:

  * Method Accuracy Comparison
  * Processing Time
  * PCA Components vs Accuracy
  * Top 5 Performing Methods

---

## Functions Overview

| Function                                                            | Purpose                                                          |
| ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `recognize_letter(image_path)`                                      | Recognizes a single letter using Tesseract OCR                   |
| `recognize_word(image_path)`                                        | Recognizes a single word                                         |
| `recognize_text(image_path)`                                        | Recognizes full text                                             |
| `apply_pca_enhancement(image_path, n_components)`                   | Applies PCA to the image and reconstructs it                     |
| `recognize_with_pca(image_path, recognition_type)`                  | Performs OCR on PCA-enhanced image                               |
| `calculate_accuracy(predicted, actual)`                             | Returns percentage similarity between predicted and ground truth |
| `benchmark_methods(image_path, ground_truth, pca_components_range)` | Tests multiple methods and PCA components                        |
| `plot_accuracy_graphs(results, save_plots=True)`                    | Creates graphs and optionally saves them                         |
| `run_complete_analysis(image_path, ground_truth)`                   | Runs full analysis pipeline and generates graphs                 |

---

## File Structure

```
TextVision_Project/
├── Project.py               # Main application file
├── README.md                # This file
├── text/                    # Example images
│   └── para2.png
├── method_accuracy_comparison.png
├── processing_time_comparison.png
├── pca_accuracy_analysis.png
└── top_performers.png
```

---

## Example Output

**Terminal Output:**

```
Original Text: Hello W0rld
PCA Enhanced: Hello World

Accuracy Original: 85.5%
Accuracy PCA: 98.2%
Processing Time: 0.452 seconds
Best Method: PCA-50 with 98.2% accuracy
```

**Graph Window Tabs:**

1. Original Methods Accuracy
2. Processing Time Comparison
3. PCA Components vs Accuracy
4. Top 5 Performing Methods

---

## Notes

* PCA enhancement improves OCR accuracy in noisy or handwritten images.
* You can tweak `n_components` to experiment with dimensionality reduction.
* Ground truth text is required for benchmarking accuracy.

---

## License

This project is **open-source** under the MIT License. You are free to use, modify, and distribute it.

---

## Authors

**Anamta Gohar**

* GitHub: [anamta-JINX](https://github.com/anamta-JINX)
* Email: anamta.gohar25@gmail.com
* Role: Developer, Designer, and Maintainer of TextVision Project
