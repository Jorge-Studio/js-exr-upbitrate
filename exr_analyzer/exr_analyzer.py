"""
EXR Viewer & Analyzer — Cinema VFX Pipeline Diagnostic Tool
A visual GUI for analyzing EXR bit depth, waveform, and quality metrics.

Usage:
    python exr_viewer.py
    python exr_viewer.py <file.exr>
"""

import sys
import os
import math
import numpy as np

# Install dependencies if needed
def ensure_deps():
    required = ['PyQt5', 'OpenEXR', 'matplotlib', 'Pillow']
    for pkg in required:
        try:
            __import__(pkg.lower().replace('-', '_').split('[')[0])
        except ImportError:
            print(f"Installing {pkg}...")
            os.system(f"{sys.executable} -m pip install {pkg}")

ensure_deps()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QScrollArea, QGroupBox, QGridLayout, QComboBox,
    QProgressBar, QSizePolicy, QDialog, QToolBar, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QCursor

import OpenEXR
import Imath
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ══════════════════════════════════════════════════════════════════════════════
# COLOR SCHEME
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'bg': '#0D0D0D',
    'bg_light': '#1A1A1A',
    'bg_card': '#141414',
    'text': '#E8E8E8',
    'text_dim': '#888888',
    'accent': '#00FF88',
    'warning': '#FFaa00',
    'error': '#FF4444',
    'red': '#FF6B6B',
    'green': '#4ECB71',
    'blue': '#4A90D9',
    'border': '#333333',
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: 'SF Pro Display', 'Segoe UI', 'Helvetica Neue', sans-serif;
}}
QLabel {{
    color: {COLORS['text']};
}}
QPushButton {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {COLORS['border']};
    border-color: {COLORS['accent']};
}}
QPushButton:pressed {{
    background-color: {COLORS['accent']};
    color: {COLORS['bg']};
}}
QPushButton#fullscreen_btn {{
    padding: 4px 8px;
    font-size: 14px;
    min-width: 30px;
}}
QTableWidget {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']};
    border-radius: 8px;
}}
QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {COLORS['border']};
}}
QHeaderView::section {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_dim']};
    padding: 8px;
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
}}
QGroupBox {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 16px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: {COLORS['text_dim']};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QDialog {{
    background-color: {COLORS['bg']};
}}
QToolBar {{
    background-color: {COLORS['bg_light']};
    border: none;
    spacing: 5px;
    padding: 5px;
}}
QSlider::groove:horizontal {{
    border: 1px solid {COLORS['border']};
    height: 6px;
    background: {COLORS['bg_light']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    border: none;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
QSpinBox {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    padding: 4px;
    border-radius: 4px;
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# EXR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_CHROMATICITIES = {
    "ACES AP0": ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.077), (0.3217, 0.3377)),
    "ACES AP1 (ACEScg)": ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), (0.3217, 0.3377)),
    "Rec.709 / sRGB": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), (0.3127, 0.3290)),
    "Rec.2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), (0.3127, 0.3290)),
    "DCI-P3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), (0.3140, 0.3510)),
}


def identify_colorspace(chrom):
    if chrom is None:
        return "Unknown"
    try:
        r = (chrom.red.x, chrom.red.y)
        g = (chrom.green.x, chrom.green.y)
        b = (chrom.blue.x, chrom.blue.y)
        w = (chrom.white.x, chrom.white.y)
        file_chrom = (r, g, b, w)
    except AttributeError:
        return "Unknown"
    
    best_match, best_dist = "Unknown", float('inf')
    for name, ref in KNOWN_CHROMATICITIES.items():
        dist = sum(abs(float(fv) - float(rv)) for fc, rc in zip(file_chrom, ref) for fv, rv in zip(fc, rc))
        if dist < best_dist:
            best_dist, best_match = dist, name
    
    if best_dist < 0.05:
        return best_match
    elif best_dist < 0.15:
        return f"{best_match} (approx)"
    return "Unknown"


def detect_channel_type(header, channel_name):
    ch_info = header['channels'].get(channel_name)
    if ch_info is None:
        return None
    ptype = ch_info.type
    if ptype == Imath.PixelType(Imath.PixelType.HALF):
        return "HALF"
    elif ptype == Imath.PixelType(Imath.PixelType.FLOAT):
        return "FLOAT"
    return "UNKNOWN"


def read_channel(exr_file, header, channel_name):
    ch_type = detect_channel_type(header, channel_name)
    if ch_type == "FLOAT":
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr_file.channel(channel_name, pt)
        arr = np.frombuffer(raw, dtype=np.float32)
        bit_label = "32-bit FLOAT"
    elif ch_type == "HALF":
        pt = Imath.PixelType(Imath.PixelType.HALF)
        raw = exr_file.channel(channel_name, pt)
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        bit_label = "16-bit HALF"
    else:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr_file.channel(channel_name, pt)
        arr = np.frombuffer(raw, dtype=np.float32)
        bit_label = f"{ch_type}"
    return arr, bit_label, ch_type


def detect_encoding(mean_vals, max_vals):
    avg_mean = np.mean(mean_vals)
    avg_max = np.mean(max_vals)
    if avg_max > 5.0 and avg_mean < 2.0:
        return "Linear (scene-referred)"
    elif avg_max < 1.5 and avg_mean > 0.15:
        return "Log (ACEScct/LogC)"
    elif avg_max < 1.1:
        return "Linear (SDR)"
    elif avg_max > 1.5:
        return "Linear (scene-referred)"
    return "Unknown"


def analyze_exr(filepath):
    """Analyze an EXR file and return all metrics."""
    f = OpenEXR.InputFile(filepath)
    header = f.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    fsize = os.path.getsize(filepath)
    
    channels = sorted(header['channels'].keys())
    rgb_channels = [ch for ch in ['R', 'G', 'B'] if ch in channels]
    
    img_data = np.zeros((height, width, 3), dtype=np.float32)
    results = {}
    means, maxes = [], []
    
    for i, ch in enumerate(rgb_channels):
        arr, bit_label, ch_type = read_channel(f, header, ch)
        arr = arr.reshape((height, width))
        img_data[:, :, i] = arr
        
        finite = arr[np.isfinite(arr)].flatten()
        unique = np.unique(finite)
        
        midtone = unique[(unique > 0.2) & (unique < 0.5)]
        if len(midtone) > 1:
            diffs = np.diff(midtone.astype(np.float64))
            eight_bit_step = 1.0 / 255
            step_ratio = eight_bit_step / diffs.mean() if diffs.mean() > 0 else 0
        else:
            step_ratio = 0
        
        results[ch] = {
            'unique_count': len(unique),
            'bit_label': bit_label,
            'ch_type': ch_type,
            'min': float(finite.min()),
            'max': float(finite.max()),
            'mean': float(finite.mean()),
            'step_ratio': step_ratio,
            'data': finite,
        }
        means.append(finite.mean())
        maxes.append(finite.max())
    
    encoding = detect_encoding(means, maxes)
    chrom = header.get('chromaticities')
    colorspace = identify_colorspace(chrom)
    compression = str(header['compression']).replace('Compression.', '')
    
    unique_counts = [results[ch]['unique_count'] for ch in rgb_channels]
    avg_unique = np.mean(unique_counts)
    eff_bits = math.log2(avg_unique) if avg_unique > 1 else 0
    
    all_vals = img_data.flatten()
    above_1 = 100 * np.sum(all_vals > 1.0) / len(all_vals)
    
    if eff_bits >= 13.0:
        rating = "★★★★★ Cinema-grade"
    elif eff_bits >= 11.5:
        rating = "★★★★☆ Good"
    elif eff_bits >= 10.0:
        rating = "★★★☆☆ Acceptable"
    elif eff_bits >= 8.5:
        rating = "★★☆☆☆ Poor"
    else:
        rating = "★☆☆☆☆ 8-bit equivalent"
    
    ch_types = set(results[ch]['ch_type'] for ch in rgb_channels)
    native_type = "32-bit float" if "FLOAT" in ch_types else "16-bit half"
    avg_step_ratio = np.mean([results[ch]['step_ratio'] for ch in rgb_channels])
    
    return {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'width': width,
        'height': height,
        'filesize': fsize,
        'filesize_mb': fsize / 1024 / 1024,
        'compression': compression,
        'native_type': native_type,
        'colorspace': colorspace,
        'encoding': encoding,
        'eff_bits': eff_bits,
        'avg_unique': avg_unique,
        'above_1_pct': above_1,
        'rating': rating,
        'results': results,
        'img_data': img_data,
        'avg_step_ratio': avg_step_ratio,
        'range_min': min(results[ch]['min'] for ch in rgb_channels),
        'range_max': max(results[ch]['max'] for ch in rgb_channels),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FULLSCREEN DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class FullscreenDialog(QDialog):
    """Fullscreen dialog for detailed visualization."""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(STYLESHEET)
        self.setMinimumSize(1200, 800)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(10, 10, 10, 10)
        
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont('SF Pro Display', 16, QFont.Bold))
        toolbar_layout.addWidget(self.title_label)
        toolbar_layout.addStretch()
        
        self.close_btn = QPushButton("✕ Close")
        self.close_btn.clicked.connect(self.close)
        toolbar_layout.addWidget(self.close_btn)
        
        layout.addWidget(self.toolbar_widget)
        
        # Content area
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MATPLOTLIB WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class InteractiveCanvas(FigureCanvas):
    """Base class for interactive matplotlib canvases with zoom/pan."""
    
    fullscreen_requested = pyqtSignal()
    
    def __init__(self, figsize=(8, 4), parent=None):
        self.fig = Figure(figsize=figsize, facecolor=COLORS['bg_card'])
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._setup_axes()
        
        # Enable mouse interaction
        self.setFocusPolicy(Qt.StrongFocus)
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)
        
        self._pan_start = None
        self._pan_xlim = None
        self._pan_ylim = None
    
    def _setup_axes(self):
        self.ax.set_facecolor(COLORS['bg'])
        self.ax.tick_params(colors=COLORS['text_dim'], labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['border'])
        self.ax.grid(True, color=COLORS['border'], alpha=0.3, linestyle='-', linewidth=0.5)
    
    def _on_scroll(self, event):
        """Zoom with scroll wheel."""
        if event.inaxes != self.ax:
            return
        
        scale = 1.2 if event.button == 'down' else 1/1.2
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        new_xlim = [xdata - (xdata - xlim[0]) * scale,
                    xdata + (xlim[1] - xdata) * scale]
        new_ylim = [ydata - (ydata - ylim[0]) * scale,
                    ydata + (ylim[1] - ydata) * scale]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw()
    
    def _on_press(self, event):
        """Start pan on middle click or right click."""
        if event.button in [2, 3] and event.inaxes == self.ax:
            self._pan_start = (event.x, event.y)
            self._pan_xlim = self.ax.get_xlim()
            self._pan_ylim = self.ax.get_ylim()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
    
    def _on_release(self, event):
        """End pan."""
        self._pan_start = None
        self.setCursor(QCursor(Qt.ArrowCursor))
    
    def _on_motion(self, event):
        """Pan while dragging."""
        if self._pan_start is None:
            return
        
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        
        # Convert pixel delta to data delta
        xlim = self._pan_xlim
        ylim = self._pan_ylim
        
        # Get axes dimensions in pixels
        bbox = self.ax.get_window_extent()
        
        dx_data = -dx * (xlim[1] - xlim[0]) / bbox.width
        dy_data = dy * (ylim[1] - ylim[0]) / bbox.height
        
        self.ax.set_xlim([xlim[0] + dx_data, xlim[1] + dx_data])
        self.ax.set_ylim([ylim[0] + dy_data, ylim[1] + dy_data])
        self.draw()
    
    def reset_view(self):
        """Reset to default view - override in subclasses."""
        pass


class WaveformWidget(InteractiveCanvas):
    """DaVinci Resolve-style waveform display with zoom/pan."""
    
    def __init__(self, parent=None, fullscreen=False):
        figsize = (12, 8) if fullscreen else (8, 4)
        super().__init__(figsize=figsize, parent=parent)
        self.img_data = None
        self.max_display = 1.1
        self.fullscreen = fullscreen
    
    def update_waveform(self, img_data, max_display=1.1):
        """Generate waveform from image data."""
        self.img_data = img_data
        self.max_display = max_display
        self._render()
    
    def _render(self):
        self.ax.clear()
        self._setup_axes()
        
        if self.img_data is None:
            self.draw()
            return
        
        h, w, c = self.img_data.shape
        
        # More samples for fullscreen
        step = max(1, w // (1000 if self.fullscreen else 400))
        
        colors_list = [COLORS['red'], COLORS['green'], COLORS['blue']]
        
        for ch_idx, color in enumerate(colors_list):
            channel = self.img_data[:, ::step, ch_idx]
            x_positions = np.linspace(0, 1, channel.shape[1])
            
            for x_idx in range(channel.shape[1]):
                col_data = channel[:, x_idx]
                x_vals = np.full_like(col_data, x_positions[x_idx])
                
                # Use smaller markers for denser display
                size = 0.3 if self.fullscreen else 0.5
                alpha = 0.15 if self.fullscreen else 0.1
                
                self.ax.scatter(x_vals, col_data, c=color, alpha=alpha, s=size, marker='.')
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, self.max_display)
        self.ax.set_ylabel('Level', color=COLORS['text_dim'], fontsize=10)
        self.ax.set_xlabel('Frame Position', color=COLORS['text_dim'], fontsize=10)
        
        # Reference lines
        for level in [0.25, 0.5, 0.75, 1.0]:
            self.ax.axhline(y=level, color=COLORS['border'], linestyle='--', linewidth=0.5, alpha=0.5)
        
        self.fig.tight_layout()
        self.draw()
    
    def reset_view(self):
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, self.max_display)
        self.draw()


class HistogramWidget(InteractiveCanvas):
    """RGB Histogram display with zoom/pan."""
    
    def __init__(self, parent=None, fullscreen=False):
        figsize = (12, 6) if fullscreen else (8, 3)
        super().__init__(figsize=figsize, parent=parent)
        self.info = None
        self.fullscreen = fullscreen
    
    def update_histogram(self, info):
        """Generate histogram from analysis info."""
        self.info = info
        self._render()
    
    def _render(self):
        self.ax.clear()
        self._setup_axes()
        
        if self.info is None:
            self.draw()
            return
        
        colors_list = [COLORS['red'], COLORS['green'], COLORS['blue']]
        channels = ['R', 'G', 'B']
        
        bins = 512 if self.fullscreen else 256
        
        for ch, color in zip(channels, colors_list):
            if ch in self.info['results']:
                data = self.info['results'][ch]['data']
                data_clipped = np.clip(data, -0.1, 2.0)
                self.ax.hist(data_clipped, bins=bins, color=color, alpha=0.5, 
                            density=True, histtype='stepfilled', label=ch)
        
        self.ax.set_xlim(-0.1, 1.5)
        self.ax.set_ylabel('Density', color=COLORS['text_dim'], fontsize=10)
        self.ax.set_xlabel('Value', color=COLORS['text_dim'], fontsize=10)
        self.ax.axvline(x=1.0, color=COLORS['warning'], linestyle='--', linewidth=1, alpha=0.7, label='1.0')
        self.ax.axvline(x=0.0, color=COLORS['text_dim'], linestyle='--', linewidth=0.5, alpha=0.5)
        
        if self.fullscreen:
            self.ax.legend(loc='upper right', facecolor=COLORS['bg_light'], edgecolor=COLORS['border'])
        
        self.fig.tight_layout()
        self.draw()
    
    def reset_view(self):
        self.ax.set_xlim(-0.1, 1.5)
        self.ax.autoscale(axis='y')
        self.draw()


class ImagePreviewWidget(InteractiveCanvas):
    """Zoomable image preview."""
    
    def __init__(self, parent=None, fullscreen=False):
        figsize = (12, 10) if fullscreen else (6, 4)
        super().__init__(figsize=figsize, parent=parent)
        self.img_data = None
        self.exposure = 0.0
        self.fullscreen = fullscreen
        self.ax.set_axis_off()
    
    def update_image(self, img_data, exposure=0.0):
        """Update preview with tone-mapped image."""
        self.img_data = img_data
        self.exposure = exposure
        self._render()
    
    def _render(self):
        self.ax.clear()
        self.ax.set_axis_off()
        
        if self.img_data is None:
            self.ax.text(0.5, 0.5, 'No image loaded', ha='center', va='center', 
                        color=COLORS['text_dim'], fontsize=12, transform=self.ax.transAxes)
            self.draw()
            return
        
        # Tone mapping
        exposure_mult = 2.0 ** self.exposure
        img = self.img_data * exposure_mult
        img = img / (1.0 + img)  # Reinhard
        img = np.power(np.clip(img, 0, 1), 1.0/2.2)  # Gamma
        
        self.ax.imshow(img, aspect='auto')
        self.ax.set_xlim(0, img.shape[1])
        self.ax.set_ylim(img.shape[0], 0)
        
        self.fig.tight_layout()
        self.draw()
    
    def reset_view(self):
        if self.img_data is not None:
            self.ax.set_xlim(0, self.img_data.shape[1])
            self.ax.set_ylim(self.img_data.shape[0], 0)
            self.draw()


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION PANEL WITH FULLSCREEN BUTTON
# ══════════════════════════════════════════════════════════════════════════════

class VisualizationPanel(QWidget):
    """Panel containing a visualization with fullscreen button."""
    
    def __init__(self, title, widget_class, parent=None):
        super().__init__(parent)
        self.title = title
        self.widget_class = widget_class
        self.current_data = None
        self.parent_window = parent
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header with title and buttons
        header = QHBoxLayout()
        
        title_label = QLabel(title.upper())
        title_label.setStyleSheet(f"""
            color: {COLORS['text_dim']}; 
            font-size: 11px; 
            font-weight: 600;
            letter-spacing: 1px;
        """)
        header.addWidget(title_label)
        header.addStretch()
        
        # Reset view button
        reset_btn = QPushButton("⟲")
        reset_btn.setObjectName("fullscreen_btn")
        reset_btn.setToolTip("Reset view")
        reset_btn.clicked.connect(self._reset_view)
        header.addWidget(reset_btn)
        
        # Fullscreen button
        fullscreen_btn = QPushButton("⛶")
        fullscreen_btn.setObjectName("fullscreen_btn")
        fullscreen_btn.setToolTip("Fullscreen")
        fullscreen_btn.clicked.connect(self._open_fullscreen)
        header.addWidget(fullscreen_btn)
        
        layout.addLayout(header)
        
        # Main widget
        self.widget = widget_class(parent=self)
        layout.addWidget(self.widget)
        
        # Controls hint
        hint = QLabel("Scroll: Zoom | Right-drag: Pan")
        hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 9px;")
        hint.setAlignment(Qt.AlignRight)
        layout.addWidget(hint)
    
    def update_data(self, *args, **kwargs):
        """Store data and update widget."""
        self.current_data = (args, kwargs)
        self._update_widget(self.widget, *args, **kwargs)
    
    def _update_widget(self, widget, *args, **kwargs):
        """Update a widget with the current data."""
        if self.widget_class == WaveformWidget:
            widget.update_waveform(*args, **kwargs)
        elif self.widget_class == HistogramWidget:
            widget.update_histogram(*args, **kwargs)
        elif self.widget_class == ImagePreviewWidget:
            widget.update_image(*args, **kwargs)
    
    def _reset_view(self):
        self.widget.reset_view()
    
    def _open_fullscreen(self):
        if self.current_data is None:
            return
        
        dialog = FullscreenDialog(self.title, self.parent_window)
        
        # Create fullscreen version of widget
        fs_widget = self.widget_class(parent=dialog, fullscreen=True)
        
        # Toolbar with controls
        controls = QHBoxLayout()
        
        reset_btn = QPushButton("⟲ Reset View")
        reset_btn.clicked.connect(fs_widget.reset_view)
        controls.addWidget(reset_btn)
        
        controls.addStretch()
        
        # Info label
        info = QLabel("Scroll: Zoom | Right-drag: Pan | Esc: Close")
        info.setStyleSheet(f"color: {COLORS['text_dim']};")
        controls.addWidget(info)
        
        dialog.content_layout.addLayout(controls)
        dialog.content_layout.addWidget(fs_widget)
        
        # Update with current data
        args, kwargs = self.current_data
        self._update_widget(fs_widget, *args, **kwargs)
        
        dialog.showMaximized()
        dialog.exec_()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class EXRViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EXR Analyzer — Cinema VFX Diagnostic Tool")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(STYLESHEET)
        
        self.current_info = None
        self.compare_info = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("EXR ANALYZER")
        title.setFont(QFont('SF Pro Display', 24, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['text']}; letter-spacing: 2px;")
        header.addWidget(title)
        header.addStretch()
        
        self.open_btn = QPushButton("Open EXR")
        self.open_btn.clicked.connect(self.open_file)
        header.addWidget(self.open_btn)
        
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.clicked.connect(self.open_compare_file)
        self.compare_btn.setEnabled(False)
        header.addWidget(self.compare_btn)
        
        main_layout.addLayout(header)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Visualizations
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        
        # Image preview panel
        self.image_panel = VisualizationPanel("Image Preview", ImagePreviewWidget, self)
        left_layout.addWidget(self.image_panel, stretch=2)
        
        # Waveform panel
        self.waveform_panel = VisualizationPanel("Waveform", WaveformWidget, self)
        left_layout.addWidget(self.waveform_panel, stretch=1)
        
        # Histogram panel
        self.histogram_panel = VisualizationPanel("Histogram", HistogramWidget, self)
        left_layout.addWidget(self.histogram_panel, stretch=1)
        
        splitter.addWidget(left_panel)
        
        # Right side: Stats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        
        # File info
        file_group = QGroupBox("FILE INFO")
        file_layout = QGridLayout(file_group)
        self.file_labels = {}
        file_fields = [
            ('Filename', 'filename'),
            ('Resolution', 'resolution'),
            ('File Size', 'filesize'),
            ('Compression', 'compression'),
            ('Bit Depth', 'native_type'),
            ('Color Space', 'colorspace'),
            ('Encoding', 'encoding'),
        ]
        for i, (label, key) in enumerate(file_fields):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {COLORS['text']}; font-size: 13px; font-weight: 500;")
            file_layout.addWidget(lbl, i, 0)
            file_layout.addWidget(val, i, 1)
            self.file_labels[key] = val
        right_layout.addWidget(file_group)
        
        # Quality metrics
        quality_group = QGroupBox("QUALITY METRICS")
        quality_layout = QGridLayout(quality_group)
        self.quality_labels = {}
        quality_fields = [
            ('Range', 'range'),
            ('Above 1.0', 'above_1'),
            ('Unique Values', 'unique'),
            ('Midtone Step', 'step_ratio'),
            ('Effective Bits', 'eff_bits'),
            ('Quality', 'rating'),
        ]
        for i, (label, key) in enumerate(quality_fields):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {COLORS['text']}; font-size: 13px; font-weight: 500;")
            quality_layout.addWidget(lbl, i, 0)
            quality_layout.addWidget(val, i, 1)
            self.quality_labels[key] = val
        right_layout.addWidget(quality_group)
        
        # Channel details
        channel_group = QGroupBox("CHANNEL ANALYSIS")
        channel_layout = QVBoxLayout(channel_group)
        self.channel_table = QTableWidget(3, 5)
        self.channel_table.setHorizontalHeaderLabels(['Channel', 'Min', 'Max', 'Mean', 'Unique'])
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.setMaximumHeight(150)
        channel_layout.addWidget(self.channel_table)
        right_layout.addWidget(channel_group)
        
        # Comparison table
        self.compare_group = QGroupBox("COMPARISON")
        compare_layout = QVBoxLayout(self.compare_group)
        self.compare_table = QTableWidget(8, 3)
        self.compare_table.setHorizontalHeaderLabels(['Metric', 'File 1', 'File 2'])
        self.compare_table.verticalHeader().setVisible(False)
        compare_layout.addWidget(self.compare_table)
        self.compare_group.hide()
        right_layout.addWidget(self.compare_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 500])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status = QLabel("Ready — Open an EXR file to analyze")
        self.status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px; padding: 8px;")
        main_layout.addWidget(self.status)
    
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open EXR File", "", "EXR Files (*.exr);;All Files (*)"
        )
        if filepath:
            self.analyze_file(filepath)
    
    def open_compare_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open EXR to Compare", "", "EXR Files (*.exr);;All Files (*)"
        )
        if filepath:
            self.compare_info = analyze_exr(filepath)
            self.update_comparison()
    
    def analyze_file(self, filepath):
        self.status.setText(f"Analyzing {os.path.basename(filepath)}...")
        QApplication.processEvents()
        
        try:
            self.current_info = analyze_exr(filepath)
            self.update_display()
            self.compare_btn.setEnabled(True)
            self.status.setText(f"✓ Analyzed: {self.current_info['filename']}")
        except Exception as e:
            self.status.setText(f"✗ Error: {str(e)}")
    
    def update_display(self):
        info = self.current_info
        if not info:
            return
        
        # Update file info
        self.file_labels['filename'].setText(info['filename'])
        self.file_labels['resolution'].setText(f"{info['width']} × {info['height']}")
        self.file_labels['filesize'].setText(f"{info['filesize_mb']:.1f} MB")
        self.file_labels['compression'].setText(info['compression'])
        self.file_labels['native_type'].setText(info['native_type'])
        self.file_labels['colorspace'].setText(info['colorspace'])
        self.file_labels['encoding'].setText(info['encoding'])
        
        # Update quality metrics
        self.quality_labels['range'].setText(f"{info['range_min']:.3f} — {info['range_max']:.3f}")
        self.quality_labels['above_1'].setText(f"{info['above_1_pct']:.1f}%")
        self.quality_labels['unique'].setText(f"{info['avg_unique']:.0f}")
        self.quality_labels['step_ratio'].setText(f"{info['avg_step_ratio']:.1f}× finer than 8-bit")
        self.quality_labels['eff_bits'].setText(f"~{info['eff_bits']:.1f} bits")
        self.quality_labels['rating'].setText(info['rating'])
        
        # Color code rating
        if "★★★★★" in info['rating']:
            color = COLORS['accent']
        elif "★★★★" in info['rating']:
            color = COLORS['accent']
        elif "★★★" in info['rating']:
            color = COLORS['warning']
        else:
            color = COLORS['error']
        self.quality_labels['rating'].setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 600;")
        
        # Update channel table
        self.channel_table.setRowCount(3)
        for i, ch in enumerate(['R', 'G', 'B']):
            if ch in info['results']:
                r = info['results'][ch]
                self.channel_table.setItem(i, 0, QTableWidgetItem(ch))
                self.channel_table.setItem(i, 1, QTableWidgetItem(f"{r['min']:.4f}"))
                self.channel_table.setItem(i, 2, QTableWidgetItem(f"{r['max']:.4f}"))
                self.channel_table.setItem(i, 3, QTableWidgetItem(f"{r['mean']:.4f}"))
                self.channel_table.setItem(i, 4, QTableWidgetItem(f"{r['unique_count']}"))
        
        # Update visualizations
        self.image_panel.update_data(info['img_data'])
        self.waveform_panel.update_data(info['img_data'])
        self.histogram_panel.update_data(info)
    
    def update_comparison(self):
        if not self.current_info or not self.compare_info:
            return
        
        self.compare_group.show()
        a, b = self.current_info, self.compare_info
        
        rows = [
            ('Filename', a['filename'], b['filename']),
            ('File Size', f"{a['filesize_mb']:.1f} MB", f"{b['filesize_mb']:.1f} MB"),
            ('Compression', a['compression'], b['compression']),
            ('Range', f"{a['range_min']:.3f} - {a['range_max']:.3f}", f"{b['range_min']:.3f} - {b['range_max']:.3f}"),
            ('Above 1.0', f"{a['above_1_pct']:.1f}%", f"{b['above_1_pct']:.1f}%"),
            ('Unique Values', f"{a['avg_unique']:.0f}", f"{b['avg_unique']:.0f}"),
            ('Midtone Step', f"{a['avg_step_ratio']:.1f}×", f"{b['avg_step_ratio']:.1f}×"),
            ('Effective Bits', f"~{a['eff_bits']:.1f}", f"~{b['eff_bits']:.1f}"),
        ]
        
        self.compare_table.setRowCount(len(rows))
        for i, (label, va, vb) in enumerate(rows):
            self.compare_table.setItem(i, 0, QTableWidgetItem(label))
            self.compare_table.setItem(i, 1, QTableWidgetItem(va))
            self.compare_table.setItem(i, 2, QTableWidgetItem(vb))
        
        diff = a['eff_bits'] - b['eff_bits']
        if abs(diff) > 0.2:
            winner_msg = f"{'File 1' if diff > 0 else 'File 2'} has ~{abs(diff):.1f} more effective bits"
            self.status.setText(f"Comparison: {winner_msg}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = EXRViewer()
    window.show()
    
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        window.analyze_file(sys.argv[1])
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
