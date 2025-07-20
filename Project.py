import cv2              
import pytesseract      
from PIL import Image   
import numpy as np      
from sklearn.decomposition import PCA       
import matplotlib.pyplot as plt             
from difflib import SequenceMatcher         
import time             

def recognize_letter(image_path):
    """
    Recognize a single letter from an image file
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Use pytesseract to extract text
        # --psm 10 treats the image as a single character
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(thresh, config=config)
        
        # Clean the result
        letter = text.strip()
        
        return letter if letter else "Could not recognize letter"
        
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_word(image_path):
    """
    Recognize a word from an image file
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Use pytesseract to extract text
        # --psm 8 treats the image as a single word
        config = '--psm 8'
        text = pytesseract.image_to_string(thresh, config=config)
        
        # Clean the result
        word = text.strip()
        
        return word if word else "Could not recognize word"
        
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_text(image_path):
    """
    Recognize any text (multiple words/lines) from an image file
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Use pytesseract to extract text
        # --psm 6 assumes uniform block of text
        config = '--psm 6'
        text = pytesseract.image_to_string(thresh, config=config)
        
        # Clean the result
        recognized_text = text.strip()
        
        return recognized_text if recognized_text else "Could not recognize text"
        
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_letter_pil(image_path):
    """
    Alternative method using PIL directly for single letters
    """
    try:
        # Open image with PIL
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Convert to numpy array for thresholding
        img_array = np.array(img)
        
        # Apply threshold
        _, thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # Convert back to PIL Image
        thresh_img = Image.fromarray(thresh)
        
        # OCR configuration for single character
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(thresh_img, config=config)
        
        return text.strip() if text.strip() else "Could not recognize letter"
        
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_word_pil(image_path):
    """
    Alternative method using PIL directly for words
    """
    try:
        # Open image with PIL
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Convert to numpy array for thresholding
        img_array = np.array(img)
        
        # Apply threshold
        _, thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # Convert back to PIL Image
        thresh_img = Image.fromarray(thresh)
        
        # OCR configuration for single word
        config = '--psm 8'
        text = pytesseract.image_to_string(thresh_img, config=config)
        
        return text.strip() if text.strip() else "Could not recognize word"
        
    except Exception as e:
        return f"Error: {str(e)}"

def apply_pca_enhancement(image_path, n_components=50):
    """
    Apply PCA to image for dimensionality reduction and enhancement
    
    Args:
        image_path (str): Path to the image file
        n_components (int): Number of principal components to keep
    
    Returns:
        numpy.ndarray: PCA-enhanced image
    """
    try:
        # Read and preprocess image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Reshape image for PCA (flatten each row)
        h, w = thresh.shape
        reshaped = thresh.reshape(h, w)
        
        # Create data matrix (each row is a feature vector)
        data = []
        for i in range(h):
            data.append(reshaped[i, :])
        data = np.array(data)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, min(h, w)))
        pca_result = pca.fit_transform(data)
        
        # Reconstruct image
        reconstructed = pca.inverse_transform(pca_result)
        reconstructed = reconstructed.reshape(h, w)
        
        # Normalize to 0-255 range
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return reconstructed, pca.explained_variance_ratio_
        
    except Exception as e:
        print(f"PCA Error: {str(e)}")
        return None, None

def recognize_with_pca(image_path, recognition_type='text'):
    """
    Recognize text using PCA-enhanced image
    
    Args:
        image_path (str): Path to the image file
        recognition_type (str): 'letter', 'word', or 'text'
    
    Returns:
        str: Recognition result with PCA enhancement
    """
    try:
        # Apply PCA enhancement
        enhanced_img, variance_ratio = apply_pca_enhancement(image_path)
        
        if enhanced_img is None:
            return "PCA enhancement failed"
        
        # Convert to PIL Image for OCR
        pil_img = Image.fromarray(enhanced_img)
        
        # Choose OCR configuration based on type
        if recognition_type == 'letter':
            config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        elif recognition_type == 'word':
            config = '--psm 8'
        else:
            config = '--psm 6'
        
        # Perform OCR
        text = pytesseract.image_to_string(pil_img, config=config)
        result = text.strip()
        
        # Calculate explained variance
        total_variance = np.sum(variance_ratio) * 100
        
        return {
            'text': result if result else f"Could not recognize {recognition_type}",
            'explained_variance': f"{total_variance:.1f}%"
        }
        
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_accuracy(predicted, actual):
    """Calculate accuracy between predicted and actual text"""
    if not predicted or not actual:
        return 0.0
    
    # Use sequence matcher for similarity
    similarity = SequenceMatcher(None, predicted.lower(), actual.lower()).ratio()
    return similarity * 100

def benchmark_methods(image_path, ground_truth, pca_components_range=None):
  
    if pca_components_range is None:
        pca_components_range = [10, 20, 30, 50, 75, 100]
    
    results = {
        'methods': [],
        'accuracies': [],
        'times': [],
        'pca_components': [],
        'pca_accuracies': []
    }
    
    # Test original methods
    methods = [
        ('Original Letter', lambda: recognize_letter(image_path)),
        ('Original Word', lambda: recognize_word(image_path)),
        ('Original Text', lambda: recognize_text(image_path)),
        ('PIL Letter', lambda: recognize_letter_pil(image_path)),
        ('PIL Word', lambda: recognize_word_pil(image_path))
    ]
    
    print("Testing original methods...")
    for method_name, method_func in methods:       
        start_time = time.time()
        result = method_func()
        end_time = time.time()
        
        accuracy = calculate_accuracy(result, ground_truth)
        
        results['methods'].append(method_name)
        results['accuracies'].append(accuracy)
        results['times'].append(end_time - start_time)
    
    # Test PCA methods with different components
    print("Testing PCA methods...")
    for n_components in pca_components_range:
        start_time = time.time()
        
        # Apply PCA with specific components
        enhanced_img, _ = apply_pca_enhancement(image_path, n_components)
        if enhanced_img is not None:
            pil_img = Image.fromarray(enhanced_img)
            config = '--psm 6'
            text = pytesseract.image_to_string(pil_img, config=config).strip()
        else:
            text = ""
        
        end_time = time.time()
        
        accuracy = calculate_accuracy(text, ground_truth)
        
        results['pca_components'].append(n_components)
        results['pca_accuracies'].append(accuracy)
    
    return results

def plot_accuracy_graphs(results, save_plots=True):
    """
    Create accuracy visualization graphs using plt syntax
    """
    plt.style.use('default')
    
    # 1. Method Accuracy Comparison
    plt.figure(figsize=(12, 8))
    methods = results['methods']
    accuracies = results['accuracies']
    
    bars = plt.bar(range(len(methods)), accuracies, color=['#FF6B6B', '#4ECDC4', "#AA2CDB", '#96CEB4', "#AFE755"])
    plt.xlabel('OCR Methods')
    plt.ylabel('Accuracy (%)')
    plt.title('OCR Method Accuracy Comparison')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('method_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Processing Time Comparison
    plt.figure(figsize=(12, 8))
    times = results['times']
    bars2 = plt.bar(range(len(methods)), times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    plt.xlabel('OCR Methods')
    plt.ylabel('Processing Time (seconds)')
    plt.title('OCR Method Processing Time')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('processing_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. PCA Components vs Accuracy
    plt.figure(figsize=(12, 8))
    pca_components = results['pca_components']
    pca_accuracies = results['pca_accuracies']
    
    plt.plot(pca_components, pca_accuracies, 'o-', linewidth=2, markersize=8, color='#E74C3C')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Accuracy (%)')
    plt.title('PCA Components vs OCR Accuracy')
    plt.grid(True, alpha=0.3)
    plt.fill_between(pca_components, pca_accuracies, alpha=0.3, color='#E74C3C')
    
    # Add best point annotation
    if pca_accuracies:
        best_idx = np.argmax(pca_accuracies)
        best_comp = pca_components[best_idx]
        best_acc = pca_accuracies[best_idx]
        plt.annotate(f'Best: {best_comp} components\n{best_acc:.1f}% accuracy',
                    xy=(best_comp, best_acc), xytext=(best_comp+10, best_acc+5),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('pca_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Overall Performance Summary
    plt.figure(figsize=(12, 8))
    all_methods = methods + [f'PCA-{comp}' for comp in pca_components]
    all_accuracies = accuracies + pca_accuracies
    
    # Find top 5 performers
    top_indices = np.argsort(all_accuracies)[-5:]
    top_methods = [all_methods[i] for i in top_indices]
    top_accuracies = [all_accuracies[i] for i in top_indices]
    
    bars4 = plt.barh(range(len(top_methods)), top_accuracies, color='#27AE60')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Methods')
    plt.title('Top 5 Performing Methods')
    plt.yticks(range(len(top_methods)), top_methods)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars4, top_accuracies):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', ha='left', va='center')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('top_performers.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_complete_analysis(image_path, ground_truth):
    """
    Run complete OCR analysis with accuracy measurements and graphs
    
    Args:
        image_path (str): Path to image file  
        ground_truth (str): Expected text output
    """
    print("=== Starting Complete OCR Analysis ===\n")
    
    # Run benchmarks
    results = benchmark_methods(image_path, ground_truth)
    
    # Display results
    print("=== Results Summary ===")
    for method, acc, time_val in zip(results['methods'], results['accuracies'], results['times']):
        print(f"{method:15} | Accuracy: {acc:5.1f}% | Time: {time_val:.3f}s")
    
    print("\n=== PCA Analysis ===")
    for comp, acc in zip(results['pca_components'], results['pca_accuracies']):
        print(f"PCA-{comp:3d} components | Accuracy: {acc:5.1f}%")
    
    # Find best method
    all_accuracies = results['accuracies'] + results['pca_accuracies']
    best_acc = max(all_accuracies)
    best_idx = all_accuracies.index(best_acc)
    
    if best_idx < len(results['methods']):
        best_method = results['methods'][best_idx]
    else:
        pca_idx = best_idx - len(results['methods'])
        best_method = f"PCA-{results['pca_components'][pca_idx]}"
    
    print(f"\n=== Best Method: {best_method} with {best_acc:.1f}% accuracy ===")
    
    # Create graphs
    plot_accuracy_graphs(results)
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your image file path and expected text
    image_path = "text/para2.png"
    ground_truth = "Hello World"  # Replace with actual expected text
    
    print("=== Quick Test ===")
    print(f"Original Text: {recognize_text(image_path)}")
    print(f"PCA Enhanced: {recognize_with_pca(image_path, 'text')}")
    
    # Graph Runner
    run_complete_analysis(image_path, ground_truth)
    
    print("\n=== Manual Accuracy Test ===")
    # Test specific method
    result = recognize_text(image_path)
    accuracy = calculate_accuracy(result, ground_truth)
    print(f"Text Recognition: '{result}'")
    print(f"Expected: '{ground_truth}'")
    print(f"Accuracy: {accuracy:.2f}%")
    
    print("\n=== Usage Instructions ===")
    print("• Set 'ground_truth' to your expected text")
