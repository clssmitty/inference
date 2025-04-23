import os
import sys
import numpy as np
import cv2
from PIL import Image
import supervision as sv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QRadioButton, QButtonGroup, QGroupBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

from inference.core.entities.requests.sam2 import Sam2PromptSet
from inference.models.sam2 import SegmentAnything2

os.environ["API_KEY"] = "rg7UwJdyjfGGwkKLL8nS"

class ImageClickLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.labels = []  # True for positive, False for negative
        self.setMinimumSize(600, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #e0e0e0; border: 1px solid gray;")
        self.image_data = None
        self.segmentation_mask = None
        self.scaled_points = []
        self.scale_factor_w = 1.0
        self.scale_factor_h = 1.0
        self.app = None  # Reference to the main application
        self.pixmap_offset = QPoint(0, 0)
        self.debug_mode = True
        self.original_pixmap = None  # Store the original pixmap
        
        # Store multiple masks
        self.saved_masks = []  # List of saved binary masks
        self.saved_mask_colors = []  # Different colors for each mask
        self.mask_colors = [
            QColor(0, 120, 255, 128),   # Blue
            QColor(255, 120, 0, 128),   # Orange
            QColor(0, 255, 0, 128),     # Green
            QColor(255, 0, 255, 128),   # Purple
            QColor(255, 255, 0, 128),   # Yellow
            QColor(0, 255, 255, 128),   # Cyan
            QColor(255, 0, 0, 128),     # Red
        ]
        
        # Polygon drawing mode
        self.polygon_mode = False
        self.polygon_points = []  # Points in the polygon (original image coordinates)
        self.scaled_polygon_points = []  # Points in display coordinates
    
    def set_app(self, app):
        self.app = app
    
    def set_image(self, image_path):
        self.image_path = image_path
        self.points = []
        self.labels = []
        self.scaled_points = []
        self.saved_masks = []
        self.saved_mask_colors = []
        self.polygon_points = []
        self.scaled_polygon_points = []
        
        pixmap = QPixmap(image_path)
        self.original_pixmap = pixmap  # Store the original pixmap
        self.image_data = Image.open(image_path).convert("RGB")
        self.original_size = pixmap.size()
        
        # Store the original image dimensions
        self.orig_width = self.image_data.width
        self.orig_height = self.image_data.height
        
        self.update_image_display()
    
    def update_image_display(self):
        """Recalculate image display based on current widget size"""
        if self.original_pixmap is None:
            return
            
        # Scale pixmap while maintaining aspect ratio
        scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        
        # Calculate the offset for centered image
        self.pixmap_offset = QPoint(
            (self.width() - scaled_pixmap.width()) // 2,
            (self.height() - scaled_pixmap.height()) // 2
        )
        
        # Calculate scaling factors between original image and displayed image
        self.scale_factor_w = self.orig_width / scaled_pixmap.width()
        self.scale_factor_h = self.orig_height / scaled_pixmap.height()
        
        if self.debug_mode:
            print(f"Widget resized: {self.width()}x{self.height()}")
            print(f"Scaled pixmap size: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
            print(f"Scale factors: w={self.scale_factor_w}, h={self.scale_factor_h}")
            print(f"Pixmap offset: {self.pixmap_offset.x()}, {self.pixmap_offset.y()}")
            
        self.setPixmap(scaled_pixmap)
        
        # Recalculate the scaled points based on new display size
        self.update_scaled_points()
    
    def update_scaled_points(self):
        """Update the scaled points based on the current scale factors"""
        # Update SAM points
        self.scaled_points = []
        for point in self.points:
            # Convert original image coordinates to current display coordinates
            scaled_x = int(point[0] / self.scale_factor_w)
            scaled_y = int(point[1] / self.scale_factor_h)
            self.scaled_points.append([scaled_x, scaled_y])
            
        # Update polygon points
        self.scaled_polygon_points = []
        for point in self.polygon_points:
            scaled_x = int(point[0] / self.scale_factor_w)
            scaled_y = int(point[1] / self.scale_factor_h)
            self.scaled_polygon_points.append([scaled_x, scaled_y])
    
    def resizeEvent(self, event):
        """Handle resize events to update image display"""
        super().resizeEvent(event)
        self.update_image_display()
    
    def add_point(self, pos, is_positive):
        """Add a point for SAM segmentation"""
        # Adjust for the pixmap offset (for centered images)
        adjusted_pos = QPoint(pos.x() - self.pixmap_offset.x(), pos.y() - self.pixmap_offset.y())
        
        # Check if the click is within the image boundaries
        if adjusted_pos.x() < 0 or adjusted_pos.y() < 0 or \
           adjusted_pos.x() >= self.pixmap().width() or adjusted_pos.y() >= self.pixmap().height():
            return  # Ignore clicks outside the image
        
        # Convert position to original image coordinates
        original_x = int(adjusted_pos.x() * self.scale_factor_w)
        original_y = int(adjusted_pos.y() * self.scale_factor_h)
        
        if self.debug_mode:
            print(f"Click at screen pos: {pos.x()}, {pos.y()}")
            print(f"Adjusted pos: {adjusted_pos.x()}, {adjusted_pos.y()}")
            print(f"Mapped to original: {original_x}, {original_y}")
            
        self.points.append([original_x, original_y])
        self.scaled_points.append([adjusted_pos.x(), adjusted_pos.y()])
        self.labels.append(is_positive)
        self.update()
    
    def add_polygon_point(self, pos):
        """Add a point to the polygon being drawn"""
        # Adjust for the pixmap offset (for centered images)
        adjusted_pos = QPoint(pos.x() - self.pixmap_offset.x(), pos.y() - self.pixmap_offset.y())
        
        # Check if the click is within the image boundaries
        if adjusted_pos.x() < 0 or adjusted_pos.y() < 0 or \
           adjusted_pos.x() >= self.pixmap().width() or adjusted_pos.y() >= self.pixmap().height():
            return  # Ignore clicks outside the image
        
        # Convert position to original image coordinates
        original_x = int(adjusted_pos.x() * self.scale_factor_w)
        original_y = int(adjusted_pos.y() * self.scale_factor_h)
        
        if self.debug_mode:
            print(f"Polygon point: {original_x}, {original_y}")
            
        self.polygon_points.append([original_x, original_y])
        self.scaled_polygon_points.append([adjusted_pos.x(), adjusted_pos.y()])
        self.update()
    
    def finish_polygon(self):
        """Complete the polygon and create a mask from it"""
        if len(self.polygon_points) < 3:
            return False  # Need at least 3 points to create a polygon
            
        # Create an empty mask with the same dimensions as the original image
        mask = np.zeros((self.orig_height, self.orig_width), dtype=np.uint8)
        
        # Convert points to numpy array for OpenCV
        points = np.array([self.polygon_points], dtype=np.int32)
        
        # Fill the polygon with 255 values using OpenCV
        cv2.fillPoly(mask, points, 255)
        
        # Convert to boolean mask for consistency with SAM masks
        bool_mask = mask > 0
        
        # Set the current segmentation mask to the polygon mask
        self.segmentation_mask = bool_mask
        
        # Immediately update the display
        self.update()
        
        return True
    
    def clear_polygon(self):
        """Clear the current polygon points"""
        self.polygon_points = []
        self.scaled_polygon_points = []
        self.update()
    
    def clear_points(self):
        self.points = []
        self.labels = []
        self.scaled_points = []
        self.segmentation_mask = None
        self.update()
    
    def set_segmentation_mask(self, mask):
        self.segmentation_mask = mask
        self.update()
    
    def save_current_mask(self):
        """Save the current segmentation mask to the saved masks list"""
        if self.segmentation_mask is not None:
            if len(self.segmentation_mask.shape) == 3:
                mask = self.segmentation_mask[0].copy()  # Copy first mask from batch
            else:
                mask = self.segmentation_mask.copy()
                
            # Get the next color in the rotation
            color_idx = len(self.saved_masks) % len(self.mask_colors)
            
            self.saved_masks.append(mask)
            self.saved_mask_colors.append(self.mask_colors[color_idx])
            
            # Clear points after saving mask but keep the mask visible
            self.points = []
            self.labels = []
            self.scaled_points = []
            self.polygon_points = []
            self.scaled_polygon_points = []
            self.update()
            return True
        return False
    
    def clear_all_masks(self):
        """Clear all saved masks"""
        self.saved_masks = []
        self.saved_mask_colors = []
        self.segmentation_mask = None
        self.update()
    
    def start_polygon_mode(self):
        """Enter polygon drawing mode"""
        self.polygon_mode = True
        self.polygon_points = []
        self.scaled_polygon_points = []
        self.update()
    
    def exit_polygon_mode(self):
        """Exit polygon drawing mode"""
        self.polygon_mode = False
        self.polygon_points = []
        self.scaled_polygon_points = []
        self.update()
    
    def mousePressEvent(self, event):
        if not self.pixmap():
            return
        
        if event.button() == Qt.LeftButton and self.app is not None:
            if self.polygon_mode:
                self.add_polygon_point(event.pos())
            else:
                is_positive = self.app.is_positive_point()
                self.add_point(event.pos(), is_positive)
                # After adding a point, clear existing current mask to ensure it gets updated
                # but keep the saved masks
                self.segmentation_mask = None
                self.app.run_segmentation()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.pixmap():
            return
            
        painter = QPainter(self)
        
        # Draw saved masks first
        for i, saved_mask in enumerate(self.saved_masks):
            # Create a semi-transparent overlay for the saved mask
            h, w = saved_mask.shape
            
            # Create transparent RGBA image
            mask_image = QImage(w, h, QImage.Format_RGBA8888)
            mask_image.fill(Qt.transparent)
            
            # Fill in the segmentation with the assigned color
            for y in range(h):
                for x in range(w):
                    if saved_mask[y, x]:
                        mask_image.setPixelColor(x, y, self.saved_mask_colors[i])
            
            # Scale the mask to match the displayed pixmap size
            pixmap_size = self.pixmap().size()
            scaled_mask = QPixmap.fromImage(mask_image).scaled(
                pixmap_size, Qt.KeepAspectRatio)
                
            # Draw the mask with the same offset as the image
            painter.drawPixmap(self.pixmap_offset, scaled_mask)
        
        # Draw current segmentation mask on top if available
        if self.segmentation_mask is not None:
            if len(self.segmentation_mask.shape) == 3:
                mask = self.segmentation_mask[0]  # For batch size of 1
            else:
                mask = self.segmentation_mask
                
            # Create a semi-transparent overlay for the mask
            if self.debug_mode:
                print(f"Mask shape: {mask.shape}")
                print(f"Original image: {self.orig_width}x{self.orig_height}")
                
            # Scale the mask to match the original image dimensions if needed
            h, w = mask.shape
            if h != self.orig_height or w != self.orig_width:
                if self.debug_mode:
                    print(f"Warning: Mask dimensions ({w}x{h}) don't match original image ({self.orig_width}x{self.orig_height})")
            
            # Create transparent RGBA image
            mask_image = QImage(w, h, QImage.Format_RGBA8888)
            mask_image.fill(Qt.transparent)
            
            # Use white color for current mask to distinguish from saved ones
            current_mask_color = QColor(255, 255, 255, 128)
            
            # Fill in the segmentation with semi-transparent white
            for y in range(h):
                for x in range(w):
                    if mask[y, x]:
                        mask_image.setPixelColor(x, y, current_mask_color)
            
            # Scale the mask to match the displayed pixmap size
            pixmap_size = self.pixmap().size()
            scaled_mask = QPixmap.fromImage(mask_image).scaled(
                pixmap_size, Qt.KeepAspectRatio)
                
            if self.debug_mode:
                print(f"Scaled mask size: {scaled_mask.width()}x{scaled_mask.height()}")
                print(f"Pixmap size: {pixmap_size.width()}x{pixmap_size.height()}")
                
            # Draw the mask with the same offset as the image
            painter.drawPixmap(self.pixmap_offset, scaled_mask)
        
        # Draw SAM points
        for i, point in enumerate(self.scaled_points):
            if self.labels[i]:  # Positive point (green)
                painter.setPen(QPen(QColor(0, 255, 0), 5))
            else:  # Negative point (red)
                painter.setPen(QPen(QColor(255, 0, 0), 5))
                
            # Draw points with the offset
            point_x = point[0] + self.pixmap_offset.x()
            point_y = point[1] + self.pixmap_offset.y()
            painter.drawPoint(QPoint(point_x, point_y))
            painter.drawEllipse(QPoint(point_x, point_y), 5, 5)
        
        # Draw polygon points and lines in polygon mode
        if self.polygon_mode and len(self.scaled_polygon_points) > 0:
            # Draw polygon points in yellow
            painter.setPen(QPen(QColor(255, 255, 0), 5))
            
            # Draw each point
            for point in self.scaled_polygon_points:
                point_x = point[0] + self.pixmap_offset.x()
                point_y = point[1] + self.pixmap_offset.y()
                painter.drawPoint(QPoint(point_x, point_y))
                painter.drawEllipse(QPoint(point_x, point_y), 5, 5)
            
            # Connect points with lines
            if len(self.scaled_polygon_points) > 1:
                painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
                
                for i in range(len(self.scaled_polygon_points) - 1):
                    start_x = self.scaled_polygon_points[i][0] + self.pixmap_offset.x()
                    start_y = self.scaled_polygon_points[i][1] + self.pixmap_offset.y()
                    end_x = self.scaled_polygon_points[i+1][0] + self.pixmap_offset.x()
                    end_y = self.scaled_polygon_points[i+1][1] + self.pixmap_offset.y()
                    
                    painter.drawLine(QPoint(start_x, start_y), QPoint(end_x, end_y))
                
                # Connect last point to first point to show the complete polygon
                if len(self.scaled_polygon_points) > 2:
                    start_x = self.scaled_polygon_points[-1][0] + self.pixmap_offset.x()
                    start_y = self.scaled_polygon_points[-1][1] + self.pixmap_offset.y()
                    end_x = self.scaled_polygon_points[0][0] + self.pixmap_offset.x()
                    end_y = self.scaled_polygon_points[0][1] + self.pixmap_offset.y()
                    
                    # Use a different style for the closing line
                    painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DotLine))
                    painter.drawLine(QPoint(start_x, start_y), QPoint(end_x, end_y))


class SAM2App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Interactive Segmentation")
        self.setMinimumSize(800, 700)
        
        # Initialize SAM2 model
        self.sam_model = None
        self.image_path = None
        self.embedding = None
        self.image_id = None
        self.img_shape = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Image display area
        self.image_label = ImageClickLabel(self)
        self.image_label.set_app(self)  # Set the app reference
        main_layout.addWidget(self.image_label)
        
        # Controls area
        controls_layout = QHBoxLayout()
        
        # File operations
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_image_btn)
        
        self.load_model_btn = QPushButton("Load SAM2 Model")
        self.load_model_btn.clicked.connect(self.load_model)
        file_layout.addWidget(self.load_model_btn)
        
        self.save_result_btn = QPushButton("Save Result")
        self.save_result_btn.clicked.connect(self.save_result)
        self.save_result_btn.setEnabled(False)
        file_layout.addWidget(self.save_result_btn)
        
        self.save_masks_btn = QPushButton("Save All Masks")
        self.save_masks_btn.clicked.connect(self.save_all_masks)
        self.save_masks_btn.setEnabled(False)
        file_layout.addWidget(self.save_masks_btn)
        
        self.load_masks_btn = QPushButton("Load Masks")
        self.load_masks_btn.clicked.connect(self.load_masks)
        self.load_masks_btn.setEnabled(False)
        file_layout.addWidget(self.load_masks_btn)
        
        file_group.setLayout(file_layout)
        controls_layout.addWidget(file_group)
        
        # Point Controls
        point_group = QGroupBox("Point Type")
        point_layout = QVBoxLayout()
        
        self.positive_radio = QRadioButton("Positive Point (green)")
        self.positive_radio.setChecked(True)
        self.negative_radio = QRadioButton("Negative Point (red)")
        
        point_layout.addWidget(self.positive_radio)
        point_layout.addWidget(self.negative_radio)
        
        point_group.setLayout(point_layout)
        controls_layout.addWidget(point_group)
        
        # Segmentation controls
        segment_group = QGroupBox("Segmentation")
        segment_layout = QVBoxLayout()
        
        self.run_segment_btn = QPushButton("Run Segmentation")
        self.run_segment_btn.clicked.connect(self.run_segmentation)
        self.run_segment_btn.setEnabled(False)
        segment_layout.addWidget(self.run_segment_btn)
        
        self.save_mask_btn = QPushButton("Save Mask")
        self.save_mask_btn.clicked.connect(self.save_current_mask)
        self.save_mask_btn.setEnabled(False)
        segment_layout.addWidget(self.save_mask_btn)
        
        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        segment_layout.addWidget(self.clear_points_btn)
        
        self.clear_all_btn = QPushButton("Clear All Masks")
        self.clear_all_btn.clicked.connect(self.clear_all_masks)
        segment_layout.addWidget(self.clear_all_btn)
        
        segment_group.setLayout(segment_layout)
        controls_layout.addWidget(segment_group)
        
        # Polygon Mask Group
        polygon_group = QGroupBox("Custom Polygon Mask")
        polygon_layout = QVBoxLayout()
        
        self.start_polygon_btn = QPushButton("Start Polygon Mask")
        self.start_polygon_btn.clicked.connect(self.start_polygon_mode)
        polygon_layout.addWidget(self.start_polygon_btn)
        
        self.finish_polygon_btn = QPushButton("Finish Polygon")
        self.finish_polygon_btn.clicked.connect(self.finish_polygon)
        self.finish_polygon_btn.setEnabled(False)
        polygon_layout.addWidget(self.finish_polygon_btn)
        
        self.cancel_polygon_btn = QPushButton("Cancel Polygon")
        self.cancel_polygon_btn.clicked.connect(self.cancel_polygon)
        self.cancel_polygon_btn.setEnabled(False)
        polygon_layout.addWidget(self.cancel_polygon_btn)
        
        polygon_group.setLayout(polygon_layout)
        controls_layout.addWidget(polygon_group)
        
        main_layout.addLayout(controls_layout)
        
        # Status label
        self.status_label = QLabel("Please load a model and an image to begin")
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def load_model(self):
        try:
            self.status_label.setText("Loading SAM2 model (this might take a while)...")
            QApplication.processEvents()
            
            # Load SAM2 model
            self.sam_model = SegmentAnything2(model_id="sam2/hiera_large")
            
            self.status_label.setText("SAM2 model loaded successfully!")
            self.run_segment_btn.setEnabled(True)
            
            if self.image_path:
                self.embed_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.status_label.setText(f"Error loading model: {str(e)}")
    
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )
        
        if file_path:
            self.image_path = file_path
            self.image_label.set_image(file_path)
            self.status_label.setText(f"Image loaded: {os.path.basename(file_path)}")
            
            # Enable polygon drawing even if model isn't loaded yet
            self.start_polygon_btn.setEnabled(True)
            
            if self.sam_model:
                self.embed_image()
    
    def embed_image(self):
        if not self.sam_model or not self.image_path:
            return
        
        try:
            self.status_label.setText("Computing image embedding...")
            QApplication.processEvents()
            
            # Compute embedding
            self.embedding, self.img_shape, self.image_id = self.sam_model.embed_image(self.image_path)
            
            self.status_label.setText("Image embedding computed. Click on the image to add points.")
            self.run_segment_btn.setEnabled(True)
            self.save_result_btn.setEnabled(True)
            self.save_mask_btn.setEnabled(True)
            self.save_masks_btn.setEnabled(True)
            self.load_masks_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to embed image: {str(e)}")
            self.status_label.setText(f"Error embedding image: {str(e)}")
    
    def is_positive_point(self):
        return self.positive_radio.isChecked()
    
    def clear_points(self):
        self.image_label.clear_points()
        self.status_label.setText("Points cleared. Click on the image to add new points.")
    
    def start_polygon_mode(self):
        """Start drawing a polygon mask"""
        self.image_label.start_polygon_mode()
        
        # Update UI state
        self.start_polygon_btn.setEnabled(False)
        self.finish_polygon_btn.setEnabled(True)
        self.cancel_polygon_btn.setEnabled(True)
        
        # Disable SAM segmentation controls while in polygon mode
        self.run_segment_btn.setEnabled(False)
        self.save_mask_btn.setEnabled(True)  # Keep enabled for saving the polygon
        
        self.status_label.setText("Polygon mode active. Click on the image to add polygon points.")
    
    def finish_polygon(self):
        """Finish the polygon and create a mask from it"""
        if self.image_label.finish_polygon():
            # Exit polygon mode
            self.image_label.polygon_mode = False
            
            # Update UI state
            self.start_polygon_btn.setEnabled(True)
            self.finish_polygon_btn.setEnabled(False)
            self.cancel_polygon_btn.setEnabled(False)
            
            # Re-enable SAM segmentation if available
            if self.sam_model and self.image_path:
                self.run_segment_btn.setEnabled(True)
            
            # Force update to display the mask immediately
            self.image_label.update()
            QApplication.processEvents()
            
            self.status_label.setText("Polygon mask created. Use 'Save Mask' to keep it.")
        else:
            QMessageBox.warning(self, "Warning", "Need at least 3 points to create a polygon mask.")
    
    def cancel_polygon(self):
        """Cancel polygon drawing mode"""
        self.image_label.exit_polygon_mode()
        
        # Update UI state
        self.start_polygon_btn.setEnabled(True)
        self.finish_polygon_btn.setEnabled(False)
        self.cancel_polygon_btn.setEnabled(False)
        
        # Re-enable SAM segmentation if available
        if self.sam_model and self.image_path:
            self.run_segment_btn.setEnabled(True)
        
        self.status_label.setText("Polygon drawing canceled.")
    
    def clear_all_masks(self):
        self.image_label.clear_all_masks()
        self.status_label.setText("All masks cleared. Click on the image to create new masks.")
    
    def save_current_mask(self):
        if self.image_label.save_current_mask():
            current_mask_count = len(self.image_label.saved_masks)
            self.status_label.setText(f"Mask saved. Total saved masks: {current_mask_count}")
            
            # If we were in polygon mode, exit it
            if self.image_label.polygon_mode:
                self.cancel_polygon()
        else:
            QMessageBox.warning(self, "Warning", "No active mask to save!")
    
    def run_segmentation(self):
        if not self.sam_model or not self.image_path or not self.embedding:
            return
        
        # Always clear the previous mask before running a new segmentation
        self.image_label.segmentation_mask = None
        
        if not self.image_label.points:
            # Run with empty prompt
            empty_prompt = Sam2PromptSet(prompts=[])
            raw_masks, scores, _ = self.sam_model.segment_image(
                self.image_path, 
                prompts=empty_prompt
            )
        else:
            # Build prompts from user clicks
            # Collect all points into a single prompt
            point_list = []
            for i, point in enumerate(self.image_label.points):
                point_list.append({
                    "x": point[0], 
                    "y": point[1], 
                    "positive": self.image_label.labels[i]
                })
            
            # Create a single prompt with all points
            prompts = [{
                "points": point_list
            }]
            
            prompt_set = Sam2PromptSet(prompts=prompts)
            
            try:
                self.status_label.setText("Running segmentation...")
                QApplication.processEvents()
                
                # Debug print
                print(f"\nRunning segmentation with points:")
                for i, point in enumerate(point_list):
                    print(f"  Point {i+1}: ({point['x']}, {point['y']}), {'positive' if point['positive'] else 'negative'}")
                print(f"Total points: {len(point_list)}")
                print(f"Prompt structure: {prompt_set.prompts}")
                
                # Force repaint to clear old mask before showing new one
                self.image_label.update()
                QApplication.processEvents()
                
                raw_masks, scores, _ = self.sam_model.segment_image(
                    image=self.image_path,
                    image_id=self.image_id,
                    prompts=prompt_set,
                    use_mask_input_cache=True
                )
                
                # Debug print mask info
                if raw_masks is not None:
                    print(f"Raw masks shape: {raw_masks.shape}")
                    print(f"Scores: {scores}")
                
                self.status_label.setText(f"Segmentation complete. Confidence: {scores[0]:.4f}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Segmentation failed: {str(e)}")
                self.status_label.setText(f"Error during segmentation: {str(e)}")
                return
        
        # Apply mask threshold
        binary_masks = raw_masks >= self.sam_model.predictor.mask_threshold
        
        # Display the segmentation result
        self.image_label.set_segmentation_mask(binary_masks)
        
        # Force update to display the new mask
        self.image_label.update()
        QApplication.processEvents()
    
    def save_result(self):
        if self.image_label.segmentation_mask is None and not self.image_label.saved_masks:
            QMessageBox.warning(self, "Warning", "No segmentation result to save!")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "", "PNG Files (*.png);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                # Create visualization with mask and points
                image = np.array(Image.open(self.image_path).convert("RGB"))
                
                # Start with a clean copy of the image
                annotated_image = image.copy()
                
                # Use supervision to create annotated image
                mask_annotator = sv.MaskAnnotator(color=sv.Color.from_hex("#3585E3"), opacity=0.5)
                
                # First add all saved masks with their respective colors
                for i, mask in enumerate(self.image_label.saved_masks):
                    # Convert QColor to hex for supervision
                    color = self.image_label.saved_mask_colors[i]
                    hex_color = f"#{color.red():02x}{color.green():02x}{color.blue():02x}"
                    
                    mask_annotator = sv.MaskAnnotator(color=sv.Color.from_hex(hex_color), opacity=0.5)
                    detections = sv.Detections(
                        xyxy=np.array([[0, 0, 100, 100]]),
                        mask=np.array([mask])
                    )
                    detections.class_id = [i]
                    annotated_image = mask_annotator.annotate(annotated_image, detections)
                
                # Then add current mask if it exists
                if self.image_label.segmentation_mask is not None:
                    mask = self.image_label.segmentation_mask
                    if len(mask.shape) == 3:
                        mask = mask[0]
                        
                    current_mask_annotator = sv.MaskAnnotator(color=sv.Color.from_hex("#FFFFFF"), opacity=0.5)
                    detections = sv.Detections(
                        xyxy=np.array([[0, 0, 100, 100]]),
                        mask=np.array([mask])
                    )
                    detections.class_id = [0]
                    annotated_image = current_mask_annotator.annotate(annotated_image, detections)
                
                # Add SAM points
                for i, point in enumerate(self.image_label.points):
                    color = sv.Color.from_hex("#00FF00") if self.image_label.labels[i] else sv.Color.from_hex("#FF0000")
                    point_annotator = sv.PointAnnotator(color=color, radius=10)
                    point_detection = sv.Detections(
                        xyxy=np.array([[point[0], point[1], point[0]+1, point[1]+1]]),
                    )
                    annotated_image = point_annotator.annotate(annotated_image, point_detection)
                
                # Add polygon points if in polygon mode
                if self.image_label.polygon_mode and self.image_label.polygon_points:
                    point_annotator = sv.PointAnnotator(color=sv.Color.from_hex("#FFFF00"), radius=10)
                    for point in self.image_label.polygon_points:
                        point_detection = sv.Detections(
                            xyxy=np.array([[point[0], point[1], point[0]+1, point[1]+1]]),
                        )
                        annotated_image = point_annotator.annotate(annotated_image, point_detection)
                
                # Save the result
                Image.fromarray(annotated_image).save(file_path)
                self.status_label.setText(f"Result saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save result: {str(e)}")
    
    def save_all_masks(self):
        """Save all masks to a numpy file that can be loaded later"""
        if not self.image_label.saved_masks:
            QMessageBox.warning(self, "Warning", "No masks to save!")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Masks", "", "NumPy Files (*.npz);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                # Current mask
                current_mask = None
                if self.image_label.segmentation_mask is not None:
                    if len(self.image_label.segmentation_mask.shape) == 3:
                        current_mask = self.image_label.segmentation_mask[0].copy()
                    else:
                        current_mask = self.image_label.segmentation_mask.copy()
                
                # Convert saved mask colors to RGB tuples for saving
                color_tuples = []
                for color in self.image_label.saved_mask_colors:
                    color_tuples.append((color.red(), color.green(), color.blue(), color.alpha()))
                
                # Save masks and related data
                np.savez_compressed(
                    file_path,
                    saved_masks=np.array(self.image_label.saved_masks),
                    saved_mask_colors=np.array(color_tuples),
                    current_mask=current_mask,
                    image_path=self.image_path
                )
                
                self.status_label.setText(f"All masks saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save masks: {str(e)}")
    
    def load_masks(self):
        """Load previously saved masks from a numpy file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Masks", "", "NumPy Files (*.npz);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # Check if the loaded masks match the current image
                saved_image_path = str(data['image_path'])
                if saved_image_path != self.image_path:
                    result = QMessageBox.question(
                        self, "Image Mismatch", 
                        f"Masks were saved for image: {os.path.basename(saved_image_path)}\n"
                        f"Current image: {os.path.basename(self.image_path)}\n\n"
                        "Load anyway?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if result == QMessageBox.No:
                        return
                
                # Clear existing masks
                self.image_label.clear_all_masks()
                
                # Load saved masks
                saved_masks = data['saved_masks']
                color_tuples = data['saved_mask_colors']
                
                # Recreate QColors from tuples
                for i, mask in enumerate(saved_masks):
                    self.image_label.saved_masks.append(mask)
                    if i < len(color_tuples):
                        r, g, b, a = color_tuples[i]
                        self.image_label.saved_mask_colors.append(QColor(r, g, b, a))
                    else:
                        # Fallback to default color if colors don't match masks
                        color_idx = i % len(self.image_label.mask_colors)
                        self.image_label.saved_mask_colors.append(self.image_label.mask_colors[color_idx])
                
                # Load current mask if available
                if 'current_mask' in data and data['current_mask'] is not None:
                    self.image_label.segmentation_mask = data['current_mask']
                
                self.image_label.update()
                self.status_label.setText(f"Loaded {len(saved_masks)} masks from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load masks: {str(e)}")
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Ensure our image display is updated when the window resizes
        if hasattr(self, 'image_label') and self.image_path:
            self.image_label.update_image_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAM2App()
    window.show()
    sys.exit(app.exec_())