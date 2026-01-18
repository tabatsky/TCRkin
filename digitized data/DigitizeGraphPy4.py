import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pandas as pd
import os

class GraphDigitizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Digitizer with Zoom")
        self.root.geometry("1200x800")
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.points = []
        self.calibration_points = []
        self.calibrated = False
        self.calibration_factor_x = 1.0
        self.calibration_factor_y = 1.0
        self.calibration_offset_x = 0.0
        self.calibration_offset_y = 0.0
        
        # Zoom variables
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_step = 0.2
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.canvas_x = 0
        self.canvas_y = 0
        self.image_on_canvas = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load image button
        ttk.Button(top_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # Calibration frame
        calibration_frame = ttk.LabelFrame(top_frame, text="Calibration")
        calibration_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(calibration_frame, text="Set Calibration Points", 
                  command=self.set_calibration_mode).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(calibration_frame, text="Calculate Calibration", 
                  command=self.calculate_calibration).pack(side=tk.LEFT, padx=(0, 10))
        
        # Digitizing frame
        digitizing_frame = ttk.LabelFrame(top_frame, text="Digitizing")
        digitizing_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(digitizing_frame, text="Start Digitizing", 
                  command=self.start_digitizing).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(digitizing_frame, text="Clear Points", 
                  command=self.clear_points).pack(side=tk.LEFT, padx=(0, 10))
        
        # Zoom frame
        zoom_frame = ttk.LabelFrame(top_frame, text="Zoom & Pan")
        zoom_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(zoom_frame, text="Zoom In", 
                  command=lambda: self.zoom_image(1)).pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Button(zoom_frame, text="Zoom Out", 
                  command=lambda: self.zoom_image(-1)).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(zoom_frame, text="Reset View", 
                  command=self.reset_view).pack(side=tk.LEFT, padx=(0, 5))
        
        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.LEFT, padx=(10, 5))
        
        # Export frame
        export_frame = ttk.LabelFrame(top_frame, text="Export")
        export_frame.pack(side=tk.LEFT)
        
        ttk.Button(export_frame, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(export_frame, text="Plot Data", 
                  command=self.plot_data).pack(side=tk.LEFT, padx=5)
        
        # Content frame (image + data table)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for image
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image display frame
        image_frame = ttk.LabelFrame(left_frame, text="Graph Image - Click and drag to pan, Use buttons to zoom")
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Right frame for data table
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))
        
        # Data table frame
        data_frame = ttk.LabelFrame(right_frame, text="Digitized Data (X, Y)")
        data_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for data table
        self.data_tree = ttk.Treeview(data_frame, columns=('X', 'Y'), show='headings', height=20)
        self.data_tree.heading('X', text='X Value')
        self.data_tree.heading('Y', text='Y Value')
        self.data_tree.column('X', width=100, anchor=tk.CENTER)
        self.data_tree.column('Y', width=100, anchor=tk.CENTER)
        
        # Scrollbar for data table
        tree_scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse events for zoom and pan
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)  # Middle mouse button
        self.canvas.bind("<ButtonPress-3>", self.start_pan)  # Right mouse button
        self.canvas.bind("<B2-Motion>", self.on_pan)  # Middle mouse drag
        self.canvas.bind("<B3-Motion>", self.on_pan)  # Right mouse drag
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)   # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)   # Linux scroll down
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Load an image to begin")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.coord_var = tk.StringVar()
        ttk.Label(status_frame, textvariable=self.coord_var).pack(side=tk.RIGHT)
        
        # Mode tracking
        self.current_mode = None  # 'calibration' or 'digitizing'
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Graph Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("Bitmap files", "*.bmp"),
                ("TIFF files", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Check if file exists and is readable
                if not os.path.exists(file_path):
                    messagebox.showerror("Error", f"File not found: {file_path}")
                    return
                
                if not os.access(file_path, os.R_OK):
                    messagebox.showerror("Error", f"File is not readable: {file_path}")
                    return
                
                # Try to read the image with OpenCV
                self.original_image = cv2.imread(file_path)
                
                # Check if image was loaded successfully
                if self.original_image is None:
                    # If OpenCV fails, try with PIL as fallback
                    try:
                        pil_image = Image.open(file_path)
                        # Convert PIL image to OpenCV format
                        self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        messagebox.showinfo("Info", "Image loaded using alternative method.")
                    except Exception as pil_error:
                        messagebox.showerror("Error", 
                                           f"Cannot read image file:\n"
                                           f"OpenCV error: File format not supported or corrupted\n"
                                           f"PIL error: {str(pil_error)}\n"
                                           f"Please check if the file is a valid image.")
                        return
                
                # Convert BGR to RGB for display
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.image_path = file_path
                
                # Reset points, calibration, and zoom
                self.points = []
                self.calibration_points = []
                self.calibrated = False
                self.current_mode = None
                self.zoom_level = 1.0
                self.canvas_x = 0
                self.canvas_y = 0
                
                # Clear data table
                self.clear_data_table()
                
                self.update_display()
                self.zoom_label.config(text=f"Zoom: {self.zoom_level*100:.0f}%")
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", 
                                   f"Failed to load image:\n{str(e)}\n"
                                   f"Please make sure the file is a valid image format.")
    
    def clear_data_table(self):
        """Clear all items from the data table"""
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
    
    def update_data_table(self):
        """Update the data table with current points"""
        self.clear_data_table()
        
        for i, (img_x, img_y) in enumerate(self.points, 1):
            if self.calibrated:
                graph_x = img_x * self.calibration_factor_x + self.calibration_offset_x
                graph_y = img_y * self.calibration_factor_y + self.calibration_offset_y
                self.data_tree.insert('', 'end', values=(f"{graph_x:.4f}", f"{graph_y:.4f}"))
    
    def get_zoomed_image(self):
        """Get the image resized according to current zoom level"""
        if self.original_image is None:
            return None
            
        # Calculate new dimensions
        height, width = self.original_image.shape[:2]
        new_width = int(width * self.zoom_level)
        new_height = int(height * self.zoom_level)
        
        # Resize the image
        resized_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Draw points on the zoomed image
        for i, (img_x, img_y) in enumerate(self.calibration_points, 1):
            # Scale points to zoom level
            scaled_x = int(img_x * self.zoom_level)
            scaled_y = int(img_y * self.zoom_level)
            cv2.circle(resized_image, (scaled_x, scaled_y), max(3, int(5 * self.zoom_level)), (255, 0, 0), -1)
            cv2.putText(resized_image, f"{i}", 
                       (scaled_x+10, scaled_y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       max(0.5, 0.7 * self.zoom_level), (255, 0, 0), 2)
        
        for i, (img_x, img_y) in enumerate(self.points, 1):
            # Scale points to zoom level
            scaled_x = int(img_x * self.zoom_level)
            scaled_y = int(img_y * self.zoom_level)
            cv2.circle(resized_image, (scaled_x, scaled_y), max(2, int(3 * self.zoom_level)), (0, 255, 0), -1)
        
        return resized_image
    
    def update_display(self):
        if self.original_image is None:
            return
        
        # Get zoomed image
        zoomed_image = self.get_zoomed_image()
        if zoomed_image is None:
            return
            
        # Convert to PIL Image
        pil_image = Image.fromarray(zoomed_image)
        
        # Convert to PhotoImage for Tkinter
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.image_on_canvas = self.canvas.create_image(self.canvas_x, self.canvas_y, anchor=tk.NW, image=self.tk_image)
        
        # Update scroll region
        self.canvas.config(scrollregion=(0, 0, pil_image.width, pil_image.height))
        
        # Update zoom label
        self.zoom_label.config(text=f"Zoom: {self.zoom_level*100:.0f}%")
    
    def zoom_image(self, direction):
        """Zoom in or out"""
        if self.original_image is None:
            return
            
        # Calculate new zoom level
        if direction > 0:  # Zoom in
            new_zoom = min(self.max_zoom, self.zoom_level * (1 + self.zoom_step))
        else:  # Zoom out
            new_zoom = max(self.min_zoom, self.zoom_level / (1 + self.zoom_step))
        
        # Only update if zoom level changed
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            self.update_display()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_image is None:
            return
            
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Scroll up
            self.zoom_image(1)
        else:  # Scroll down
            self.zoom_image(-1)
    
    def start_pan(self, event):
        """Start panning the image"""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.scan_mark(event.x, event.y)
    
    def on_pan(self, event):
        """Pan the image"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def reset_view(self):
        """Reset zoom and pan to default"""
        if self.original_image is None:
            return
            
        self.zoom_level = 1.0
        self.canvas_x = 0
        self.canvas_y = 0
        self.canvas.scan_dragto(0, 0, gain=1)
        self.update_display()
        self.status_var.set("View reset to default.")
    
    def set_calibration_mode(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        
        self.current_mode = 'calibration'
        self.calibration_points = []
        self.update_display()
        self.status_var.set("Click on two known points on the graph for calibration. Use zoom for accuracy.")
    
    def calculate_calibration(self):
        if len(self.calibration_points) < 2:
            messagebox.showerror("Error", "Please set at least two calibration points.")
            return
        
        # Get calibration values from user
        calibration_window = tk.Toplevel(self.root)
        calibration_window.title("Calibration Values")
        calibration_window.geometry("300x200")
        calibration_window.transient(self.root)
        calibration_window.grab_set()
        
        ttk.Label(calibration_window, text="Enter known values for calibration points:").pack(pady=10)
        
        # Frame for point 1
        point1_frame = ttk.Frame(calibration_window)
        point1_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(point1_frame, text="Point 1 (X,Y):").pack(side=tk.LEFT)
        point1_x = ttk.Entry(point1_frame, width=10)
        point1_x.pack(side=tk.LEFT, padx=5)
        point1_y = ttk.Entry(point1_frame, width=10)
        point1_y.pack(side=tk.LEFT, padx=5)
        
        # Frame for point 2
        point2_frame = ttk.Frame(calibration_window)
        point2_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(point2_frame, text="Point 2 (X,Y):").pack(side=tk.LEFT)
        point2_x = ttk.Entry(point2_frame, width=10)
        point2_x.pack(side=tk.LEFT, padx=5)
        point2_y = ttk.Entry(point2_frame, width=10)
        point2_y.pack(side=tk.LEFT, padx=5)
        
        def apply_calibration():
            try:
                # Get values from entries
                x1 = float(point1_x.get())
                y1 = float(point1_y.get())
                x2 = float(point2_x.get())
                y2 = float(point2_y.get())
                
                # Calculate calibration factors
                img_x1, img_y1 = self.calibration_points[0]
                img_x2, img_y2 = self.calibration_points[1]
                
                # Calculate scale factors
                self.calibration_factor_x = (x2 - x1) / (img_x2 - img_x1)
                self.calibration_factor_y = (y2 - y1) / (img_y2 - img_y1)
                
                # Calculate offsets
                self.calibration_offset_x = x1 - (img_x1 * self.calibration_factor_x)
                self.calibration_offset_y = y1 - (img_y1 * self.calibration_factor_y)
                
                self.calibrated = True
                calibration_window.destroy()
                self.status_var.set("Calibration completed. You can now digitize points.")
                
                # Update data table if we have points
                if self.points:
                    self.update_data_table()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")
        
        ttk.Button(calibration_window, text="Apply Calibration", 
                  command=apply_calibration).pack(pady=20)
    
    def start_digitizing(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        
        if not self.calibrated:
            messagebox.showerror("Error", "Please calibrate the graph first.")
            return
        
        self.current_mode = 'digitizing'
        self.points = []
        self.clear_data_table()
        self.update_display()
        self.status_var.set("Click on points along the graph to digitize them. Use zoom for accuracy.")
    
    def clear_points(self):
        self.points = []
        self.clear_data_table()
        self.update_display()
        self.status_var.set("Points cleared.")
    
    def on_canvas_click(self, event):
        if self.original_image is None or self.current_mode is None:
            return
        
        # Get image coordinates (considering scroll position and zoom)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Convert to original image coordinates (accounting for zoom)
        img_x = int(x / self.zoom_level)
        img_y = int(y / self.zoom_level)
        
        if self.current_mode == 'calibration':
            # Add calibration point
            self.calibration_points.append((img_x, img_y))
            
            self.update_display()
            self.status_var.set(f"Calibration point {len(self.calibration_points)} set.")
            
        elif self.current_mode == 'digitizing' and self.calibrated:
            # Add digitized point
            self.points.append((img_x, img_y))
            
            # Convert to graph coordinates
            graph_x = img_x * self.calibration_factor_x + self.calibration_offset_x
            graph_y = img_y * self.calibration_factor_y + self.calibration_offset_y
            
            self.update_display()
            self.update_data_table()  # Update the data table
            self.status_var.set(f"Point {len(self.points)}: X={graph_x:.4f}, Y={graph_y:.4f}")
    
    def on_canvas_motion(self, event):
        if self.original_image is None:
            return
        
        # Get image coordinates (considering scroll position and zoom)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Convert to original image coordinates (accounting for zoom)
        img_x = int(x / self.zoom_level)
        img_y = int(y / self.zoom_level)
        
        if self.calibrated:
            # Convert to graph coordinates
            graph_x = img_x * self.calibration_factor_x + self.calibration_offset_x
            graph_y = img_y * self.calibration_factor_y + self.calibration_offset_y
            self.coord_var.set(f"Image: ({img_x}, {img_y})  Graph: ({graph_x:.4f}, {graph_y:.4f})  Zoom: {self.zoom_level*100:.0f}%")
        else:
            self.coord_var.set(f"Image: ({img_x}, {img_y})  Zoom: {self.zoom_level*100:.0f}%")
    
    def export_data(self):
        if not self.points:
            messagebox.showerror("Error", "No points to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Data As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            # Prepare data in two columns
            x_values = []
            y_values = []
            for img_x, img_y in self.points:
                graph_x = img_x * self.calibration_factor_x + self.calibration_offset_x
                graph_y = img_y * self.calibration_factor_y + self.calibration_offset_y
                x_values.append(graph_x)
                y_values.append(graph_y)
            
            # Create DataFrame with two columns
            df = pd.DataFrame({
                'X': x_values,
                'Y': y_values
            })
            
            # Save to file
            df.to_csv(file_path, index=False)
            self.status_var.set(f"Data exported to {os.path.basename(file_path)}")
            
            # Show confirmation with sample data
            sample_size = min(5, len(x_values))
            sample_text = "\nFirst few data points:\n"
            sample_text += "X\t\tY\n"
            sample_text += "-" * 20 + "\n"
            for i in range(sample_size):
                sample_text += f"{x_values[i]:.4f}\t{y_values[i]:.4f}\n"
            
            messagebox.showinfo("Export Successful", 
                              f"Data exported successfully!\n\n"
                              f"Total points: {len(x_values)}\n"
                              f"{sample_text}")
    
    def plot_data(self):
        if not self.points:
            messagebox.showerror("Error", "No points to plot.")
            return
        
        # Prepare data in two columns
        x_values = []
        y_values = []
        for img_x, img_y in self.points:
            graph_x = img_x * self.calibration_factor_x + self.calibration_offset_x
            graph_y = img_y * self.calibration_factor_y + self.calibration_offset_y
            x_values.append(graph_x)
            y_values.append(graph_y)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'bo-', linewidth=2, markersize=4)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Digitized Graph Data')
        plt.grid(True)
        
        # Add data table to the plot
        if len(x_values) <= 15:  # Only show table if not too many points
            table_data = [[f"{x:.4f}", f"{y:.4f}"] for x, y in zip(x_values, y_values)]
            plt.table(cellText=table_data, colLabels=['X', 'Y'], 
                     cellLoc='center', loc='bottom', bbox=[0.1, -0.5, 0.8, 0.3])
            plt.subplots_adjust(bottom=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.status_var.set("Data plotted.")

def main():
    root = tk.Tk()
    app = GraphDigitizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()