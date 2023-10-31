import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from detect import CancerDetector

class CancerPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Detection and Classification")
        self.root.geometry("900x500")

        # Create a canvas to display images
        self.img_canvas = tk.Canvas(root, bg="grey", width=340, height=200)
        self.img_canvas.place(x=90, y=60)

        # Upload button
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_and_display, bg = "sky blue")
        self.upload_btn.place(x=170, y=300, width = 200, height = 40)

        # Predict button
        self.predict_btn = tk.Button(root, text="Predict", state=tk.DISABLED, command=self.predict_cancer, bg = "light green")
        self.predict_btn.place(x=530, y=300, width = 200, height = 40)

        # Clear & Exit buttons
        self.clear_btn = tk.Button(root, text="Clear", command=self.clear_all, bg = "yellow")
        self.clear_btn.place(x=350, y=370, width = 200, height = 40)

        self.exit_btn = tk.Button(root, text="Exit", command=root.quit, bg = "red")
        self.exit_btn.place(x=350, y=430, width = 200, height = 40)

        with open('accuracy.txt', 'r') as file:
                accuracy_str = file.read()

        # Convert to percentage and round to two decimal places
        accuracy_val = float(accuracy_str)
        accuracy_percent = round(accuracy_val * 100, 2)

        # Convert to string and append '%'
        accuracy_formatted = f"{accuracy_percent}%"

        # Fixed labels
        y_position = 80

        tk.Label(root, text="Model Accuracy:").place(x=460, y=y_position)
        tk.Label(root, text=accuracy_formatted).place(x=570, y=y_position)

        # Text fields and labels
        labels = ["Morphology::", "Area size of tumor:", "Cancer stage:"]
        self.text_fields = []

        y_position += 40

        for label_text in labels:
            tk.Label(root, text=label_text).place(x=460, y=y_position)
            text_widget = tk.Text(root, height=1, width=30)
            text_widget.place(x=570, y=y_position)
            self.text_fields.append(text_widget)
            y_position += 40

        self.image_path = None


    def upload_and_display(self):
        file_types = [("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")]
        self.image_path = filedialog.askopenfilename(title="Choose an image", filetypes=file_types)
        if self.image_path:
            img = Image.open(self.image_path)
            img_resized = img.resize((340, 200))        
            img_tk = ImageTk.PhotoImage(img_resized)        
            self.img_canvas.delete("all")        
            self.img_canvas.create_image(170, 100, anchor=tk.CENTER, image=img_tk)
            self.img_canvas.image = img_tk
            self.predict_btn.config(state=tk.NORMAL)


    def predict_cancer(self):
        for field in self.text_fields:
            field.delete(1.0, tk.END)

        if self.image_path:
            detector = CancerDetector('lung_cancer_model.h5')
            prediction = detector.predict(self.image_path)

            prediction_formatted = round(prediction[1]*100, 2)
  
            self.text_fields[0].insert(tk.END, prediction[0])
            self.text_fields[1].insert(tk.END, f"{prediction_formatted} sqcm")
            self.text_fields[2].insert(tk.END, prediction[2])    


    def clear_all(self):
        self.predict_btn.config(state=tk.DISABLED)
        self.img_canvas.delete(tk.ALL)
        for field in self.text_fields:
            field.delete(1.0, tk.END)


root = tk.Tk()
app = CancerPredictionApp(root)
root.mainloop()