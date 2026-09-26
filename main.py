import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from numba import njit
import torch

from data import FastSpadLoader

class SpadScrubberApp(QMainWindow):
    def __init__(self, path_to_cube, demosaic: bool = False, rotation: int = 0, bpliteral: str = 'rggb'):
        super().__init__()
        self.setWindowTitle("Fast SPAD Cube Visualizer")
        self.setGeometry(100, 100, 1100, 750)

        self.loader = FastSpadLoader(
            spad_path, 
            rotation=rotation, 
            demosaic=demosaic,
            bayer_pattern=bpliteral
        )
        self.total_frames = self.loader.total_frames
        self.current_frame_idx = 0
        self.average_window = 1
        print(f"Cube of shape {self.loader.height}x{self.loader.width} and {self.total_frames} total frames loaded.")

        # Application state
        self.is_playing = False
        self.playback_fps = 30  # Targets standard 30 FPS playback rate
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_frame)
        self.current_frame_idx = 0
        self.average_window = 1  # Number of frames to average together

        # 2. Main Window Layout Setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        view_layout = QHBoxLayout()
        main_layout.addLayout(view_layout)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        view_layout.addWidget(self.graphics_layout, stretch=4) # Uses 80% of width

        # Create an image view plot
        self.plot_item = self.graphics_layout.addPlot()
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        self.plot_item.setAspectLocked(True)

         # 4. Hardware Histogram & Contrast Widget (Right Side)
        self.hist_widget = pg.HistogramLUTWidget()
        # Bind the histogram control handles straight to our image rendering pipeline
        self.hist_widget.setImageItem(self.image_item)
        view_layout.addWidget(self.hist_widget, stretch=1) # Uses 20% of width

        # 4. Controls Layout (Bottom Panels)
        controls_layout = QVBoxLayout()
        
        # Playback Row (Button + Scrubber Slider)
        play_row = QHBoxLayout()
        
        self.play_button = QPushButton("▶ Play")
        self.play_button.setFixedWidth(80)
        self.play_button.clicked.connect(self.toggle_playback)
        play_row.addWidget(self.play_button)
        
        self.scrub_label = QLabel("Frame: 0")
        self.scrub_label.setFixedWidth(80)
        play_row.addWidget(self.scrub_label)
        self.scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.scrub_slider.setRange(0, self.total_frames - 1)
        self.scrub_slider.valueChanged.connect(self.on_scrub_changed)
        play_row.addWidget(self.scrub_label)
        play_row.addWidget(self.scrub_slider)
        controls_layout.addLayout(play_row)

        # Averaging Window Slider
        avg_row = QHBoxLayout()
        self.avg_label = QLabel("Averaging Window: 1 frame")
        self.avg_slider = QSlider(Qt.Orientation.Horizontal)
        self.avg_slider.setRange(1, 50)
        self.avg_slider.valueChanged.connect(self.on_avg_changed)
        avg_row.addWidget(self.avg_label)
        avg_row.addWidget(self.avg_slider)
        
        controls_layout.addLayout(avg_row)
        main_layout.addLayout(controls_layout)

        self.image_item.setLevels([0, 31])
        # Render initial frame
        self.update_display()


    def toggle_playback(self):
        """Starts or pauses the loop cycle."""
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("▶ Play")
            self.is_playing = False
        else:
            # Calculate loop interval in milliseconds (e.g., 1000ms / 30fps = ~33ms)
            interval = int(1000 / self.playback_fps)
            self.timer.start(interval)
            self.play_button.setText("⏸ Pause")
            self.is_playing = True


    def advance_frame(self):
        """Timer callback loop that increments the current frame tracker."""
        next_frame = self.current_frame_idx + 1
        
        # Loop smoothly back to index zero if the end of the cube is hit
        if next_frame >= self.total_frames:
            next_frame = 0
            
        # Move slider handle (this automatically fires on_scrub_changed)
        self.scrub_slider.setValue(next_frame)


    def on_scrub_changed(self, value):
        self.current_frame_idx = value
        self.scrub_label.setText(f"Frame: {value}")
        self.update_display()


    def on_avg_changed(self, value):
        self.average_window = value
        self.avg_label.setText(f"Averaging Window: {value} frames")
        self.update_display()


    def update_display(self):
        """Requests data from the loader and shoves it to the hardware view."""
        try:
            # Let the class handle the slicing/averaging internally
            frame_to_display = self.loader.get_averaged_frame(
                self.current_frame_idx, 
                self.average_window
            )
            
            # Push the 1024x1024 frame matrix directly to your PyQtGraph ImageItem
            self.image_item.setImage(frame_to_display, autoLevels=False)
            
        except Exception as e:
            print(f"UI Update Loop Failed: {e}")


if __name__ == "__main__":
    spad_path = "/media/agarg54/ExtremeSSD/ubicam_afterlight/luther.0007/data.ubi.h5"
    
    app = QApplication(sys.argv)
    window = SpadScrubberApp(spad_path, demosaic=True, rotation=270, bpliteral='rgbg')
    window.show()
    
    sys.exit(app.exec())