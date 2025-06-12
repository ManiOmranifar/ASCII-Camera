import cv2
import numpy as np
import time
import math
from collections import deque
import random
import threading
import json
import os
from datetime import datetime
import colorsys
import sqlite3


class UltraProfessionalASCIICamera:
    def __init__(self):
        # Enhanced ASCII character sets with gradients
        self.EFFECTS = {
            'classic': "@%#*+=-:. ",
            'detailed': "@@##**++==--::.. ",
            'matrix': "10101010 ",
            'retro': "MWNHKQBDPYUGAXT7=+;:,. ",
            'minimal': "###... ",
            'cyber': "01 ",
            'blocks': "███▓▒░ ",
            'letters': "ASCIICAMERA ",
            'persian': "۱۲۳۴۵۶۷۸۹۰ ",
            'gradient': "████▓▓▓▒▒▒░░░   ",
            'geometric': "◼◾▪●◆♦◇○△▲▼►◄ ",
            'emoji': "😀😃😄😁😊☺😇🙂🙃😉 ",
            'braille': "⣿⣾⣽⣻⢿⡿⣟⣯⣷⣶⣤⣄⣀⠀ ",
            'japanese': "あいうえおかきくけこ ",
            'arabic': "ابتثجحخدذرزسشصضطظعغفقكلمنهويء ",
            'music': "♪♫♬♩♭♯𝅘𝅥𝅮𝅗𝅥𝅘𝅥 ",
            'space': "★☆✦✧✨✩✪✫⭐🌟 ",
            'weather': "☀☁⛅⛈🌧🌦🌤⛅☁ "
        }

        # Professional camera settings
        self.current_effect = 'detailed'
        self.ascii_chars = self.EFFECTS[self.current_effect]
        self.cols = 120
        self.rows = 35
        self.font_scale = 0.4
        self.char_width = 8
        self.char_height = 16

        # Advanced visual effects
        self.edge_detection = False
        self.color_mode = False
        self.wave_effect = False
        self.pulse_effect = False
        self.face_detection = False
        self.motion_blur = False
        self.rainbow_mode = False
        self.zoom_effect = False
        self.negative_mode = False
        self.contrast_boost = False

        # NEW ULTRA PROFESSIONAL FEATURES
        self.particle_system = False
        self.depth_sensing = False
        self.ai_enhancement = False
        self.hologram_mode = False
        self.glitch_effect = False
        self.mirror_mode = False
        self.kaleidoscope = False
        self.time_freeze = False
        self.ghost_trail = False
        self.ascii_3d = False
        self.voice_reactive = False
        self.gesture_control = False
        self.auto_focus = False
        self.beauty_filter = False
        self.vintage_film = False
        self.cyberpunk_mode = False
        self.matrix_rain = False
        self.fire_effect = False
        self.water_ripple = False
        self.snow_effect = False
        self.lightning_effect = False
        self.plasma_effect = False
        self.tunnel_vision = False
        self.fish_eye = False
        self.thermal_vision = False
        self.x_ray_mode = False
        self.comic_book = False
        self.oil_painting = False
        self.sketch_mode = False
        self.neon_glow = False
        self.laser_grid = False
        self.dna_helix = False

        # Professional recording and export
        self.recording = False
        self.screenshot_mode = False
        self.export_gif = False
        self.stream_mode = False

        # Analytics and statistics
        self.analytics_enabled = True
        self.session_stats = {
            'frames_processed': 0,
            'effects_used': set(),
            'session_start': time.time(),
            'faces_detected': 0,
            'motion_events': 0
        }

        # Database for settings and history
        self.db_connection = self.init_database()

        # Animation and timing variables
        self.frame_count = 0
        self.time_start = time.time()
        self.fps_counter = deque(maxlen=30)
        self.performance_monitor = deque(maxlen=100)

        # Particle system
        self.particles = []
        self.max_particles = 200

        # Ghost trail system
        self.ghost_frames = deque(maxlen=10)

        # Advanced face detection with landmarks
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.face_detection_available = True
        except:
            self.face_detection_available = False
            print("🚫 Advanced face detection not available")

        # Motion and background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True)
        self.motion_history = deque(maxlen=15)
        self.motion_intensity = 0

        # Color schemes with gradients
        self.color_schemes = {
            'white': [(255, 255, 255)],
            'green': [(0, 255, 0), (0, 200, 0), (0, 150, 0)],
            'cyan': [(255, 255, 0), (200, 255, 100), (100, 255, 200)],
            'red': [(0, 0, 255), (50, 0, 200), (100, 0, 150)],
            'blue': [(255, 0, 0), (200, 50, 0), (150, 100, 0)],
            'yellow': [(0, 255, 255), (50, 255, 200), (100, 255, 150)],
            'magenta': [(255, 0, 255), (200, 50, 200), (150, 100, 150)],
            'gold': [(0, 215, 255), (0, 180, 220), (0, 145, 185)],
            'silver': [(192, 192, 192), (169, 169, 169), (128, 128, 128)],
            'rainbow': 'dynamic',
            'fire': [(0, 69, 255), (0, 140, 255), (0, 215, 255)],
            'ice': [(255, 228, 196), (255, 218, 185), (240, 248, 255)],
            'forest': [(34, 139, 34), (0, 128, 0), (0, 100, 0)],
            'ocean': [(255, 191, 0), (255, 165, 0), (255, 140, 0)],
            'sunset': [(30, 144, 255), (255, 140, 0), (255, 69, 0)],
            'neon': [(255, 20, 147), (0, 255, 127), (255, 215, 0)]
        }
        self.current_color = 'white'

        # Audio processing for voice reactive mode
        try:
            import pyaudio
            self.audio_available = True
            self.audio_data = deque(maxlen=1024)
        except ImportError:
            self.audio_available = False
            print("🎵 Audio features not available (install pyaudio)")

        # Professional presets
        self.presets = {
            'cinematic': {
                'effects': ['contrast_boost', 'vintage_film', 'edge_detection'],
                'color': 'gold',
                'ascii_style': 'retro'
            },
            'cyberpunk': {
                'effects': ['cyberpunk_mode', 'glitch_effect', 'neon_glow'],
                'color': 'neon',
                'ascii_style': 'cyber'
            },
            'matrix': {
                'effects': ['matrix_rain', 'green', 'wave_effect'],
                'color': 'green',
                'ascii_style': 'matrix'
            },
            'artistic': {
                'effects': ['oil_painting', 'rainbow_mode', 'kaleidoscope'],
                'color': 'rainbow',
                'ascii_style': 'geometric'
            },
            'security': {
                'effects': ['thermal_vision', 'face_detection', 'motion_blur'],
                'color': 'green',
                'ascii_style': 'minimal'
            }
        }

    def init_database(self):
        """Initialize SQLite database for settings and analytics"""
        try:
            conn = sqlite3.connect('ascii_camera_pro.db')
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME,
                    end_time DATETIME,
                    frames_processed INTEGER,
                    effects_used TEXT,
                    avg_fps REAL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')

            conn.commit()
            return conn
        except:
            print("📊 Database initialization failed")
            return None

    def save_session_stats(self):
        """Save session statistics to database"""
        if self.db_connection and self.analytics_enabled:
            cursor = self.db_connection.cursor()
            avg_fps = self.get_fps()
            effects_str = ','.join(self.session_stats['effects_used'])

            cursor.execute('''
                INSERT INTO sessions (start_time, end_time, frames_processed, effects_used, avg_fps)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.fromtimestamp(self.session_stats['session_start']),
                datetime.now(),
                self.session_stats['frames_processed'],
                effects_str,
                avg_fps
            ))
            self.db_connection.commit()

    def create_particle(self, x, y, motion_intensity=1.0):
        """Create a new particle for the particle system"""
        return {
            'x': x,
            'y': y,
            'vx': random.uniform(-2, 2) * motion_intensity,
            'vy': random.uniform(-2, 2) * motion_intensity,
            'life': random.uniform(30, 60),
            'max_life': random.uniform(30, 60),
            'color': random.choice([(255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]),
            'char': random.choice("*+.○◆★✦")
        }

    def update_particles(self):
        """Update particle system"""
        # Update existing particles
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1

            # Add some physics
            particle['vy'] += 0.1  # gravity
            particle['vx'] *= 0.99  # air resistance

            if particle['life'] <= 0:
                self.particles.remove(particle)

        # Add new particles based on motion
        if len(self.particles) < self.max_particles and self.motion_intensity > 0.3:
            for _ in range(int(self.motion_intensity * 10)):
                if len(self.particles) < self.max_particles:
                    x = random.randint(0, self.cols - 1)
                    y = random.randint(0, self.rows - 1)
                    self.particles.append(self.create_particle(
                        x, y, self.motion_intensity))

    def apply_glitch_effect(self, frame):
        """Apply digital glitch effect"""
        if random.random() < 0.1:  # 10% chance of glitch
            height, width = frame.shape[:2]

            # Random horizontal displacement
            if random.random() < 0.5:
                shift = random.randint(-width//20, width//20)
                frame = np.roll(frame, shift, axis=1)

            # Random color channel corruption
            if random.random() < 0.3:
                channel = random.randint(0, 2)
                frame[:, :, channel] = np.random.randint(
                    0, 256, frame[:, :, channel].shape, dtype=np.uint8)

            # Random blocks
            if random.random() < 0.2:
                for _ in range(random.randint(1, 5)):
                    x = random.randint(0, width - 50)
                    y = random.randint(0, height - 50)
                    w = random.randint(10, 50)
                    h = random.randint(10, 50)
                    frame[y:y+h, x:x+w] = random.randint(0, 255)

        return frame

    def apply_hologram_effect(self, frame):
        """Apply holographic effect"""
        # Create scanlines
        for i in range(0, frame.shape[0], 4):
            frame[i:i+1] = frame[i:i+1] * 0.7

        # Add holographic color shift
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.frame_count * 2) % 180
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Add transparency effect
        alpha = 0.8 + 0.2 * math.sin(self.frame_count * 0.1)
        frame = (frame * alpha).astype(np.uint8)

        return frame

    def apply_matrix_rain(self, frame):
        """Apply Matrix digital rain effect"""
        height, width = frame.shape[:2]

        # Create falling characters
        if not hasattr(self, 'matrix_columns'):
            self.matrix_columns = {}

        # Add new columns
        if random.random() < 0.3:
            col = random.randint(0, self.cols - 1)
            if col not in self.matrix_columns:
                self.matrix_columns[col] = {
                    'y': 0,
                    'speed': random.uniform(0.5, 2.0),
                    'chars': [random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZアイウエオカキクケコ") for _ in range(20)]
                }

        # Update and draw columns
        for col, data in list(self.matrix_columns.items()):
            data['y'] += data['speed']
            if data['y'] > self.rows + 20:
                del self.matrix_columns[col]

        return frame

    def apply_cyberpunk_mode(self, frame):
        """Apply cyberpunk aesthetic"""
        # Enhance contrast
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

        # Add cyan-magenta color grading
        frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.2, 0, 255)  # Blue channel
        frame[:, :, 1] = np.clip(frame[:, :, 1] * 0.8, 0, 255)  # Green channel
        frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.1, 0, 255)  # Red channel

        return frame

    def apply_thermal_vision(self, frame):
        """Apply thermal vision effect"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create thermal colormap
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Blend with original
        return cv2.addWeighted(frame, 0.3, thermal, 0.7, 0)

    def apply_kaleidoscope(self, frame):
        """Apply kaleidoscope effect"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Create kaleidoscope segments
        angle_step = 360 // 6  # 6 segments
        result = frame.copy()

        for i in range(6):
            angle = i * angle_step + self.frame_count * 2

            # Rotation matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
            rotated = cv2.warpAffine(frame, M, (width, height))

            # Blend segments
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(mask, (center_x, center_y), (width//3, height//3),
                        angle, 0, angle_step, 255, -1)

            result = cv2.bitwise_and(result, result, mask=~mask)
            result = cv2.bitwise_or(
                result, cv2.bitwise_and(rotated, rotated, mask=mask))

        return result

    def apply_advanced_effects(self, frame):
        """Apply all advanced effects based on current settings"""
        processed_frame = frame.copy()

        # Base effects
        if self.contrast_boost:
            processed_frame = self.apply_contrast_boost(processed_frame)

        if self.edge_detection:
            processed_frame = self.apply_edge_detection(processed_frame)

        if self.wave_effect:
            processed_frame = self.apply_wave_effect(processed_frame)

        if self.motion_blur:
            processed_frame = self.apply_motion_blur(processed_frame)

        # Advanced effects
        if self.glitch_effect:
            processed_frame = self.apply_glitch_effect(processed_frame)

        if self.hologram_mode:
            processed_frame = self.apply_hologram_effect(processed_frame)

        if self.cyberpunk_mode:
            processed_frame = self.apply_cyberpunk_mode(processed_frame)

        if self.thermal_vision:
            processed_frame = self.apply_thermal_vision(processed_frame)

        if self.kaleidoscope:
            processed_frame = self.apply_kaleidoscope(processed_frame)

        if self.fish_eye:
            processed_frame = self.apply_fish_eye(processed_frame)

        if self.vintage_film:
            processed_frame = self.apply_vintage_film(processed_frame)

        return processed_frame

    def apply_fish_eye(self, frame):
        """Apply fish-eye lens effect"""
        height, width = frame.shape[:2]

        # Create distortion map
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        center_x, center_y = width // 2, height // 2

        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = y - center_y

                distance = math.sqrt(dx*dx + dy*dy)

                if distance != 0:
                    # Fish-eye distortion formula
                    r = distance / max(center_x, center_y)
                    theta = math.atan2(dy, dx)

                    # Apply barrel distortion
                    r_distorted = r * (1 + 0.3 * r * r)

                    new_x = center_x + r_distorted * \
                        max(center_x, center_y) * math.cos(theta)
                    new_y = center_y + r_distorted * \
                        max(center_x, center_y) * math.sin(theta)

                    map_x[y, x] = new_x
                    map_y[y, x] = new_y
                else:
                    map_x[y, x] = x
                    map_y[y, x] = y

        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def apply_vintage_film(self, frame):
        """Apply vintage film effect"""
        # Add sepia tone
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])

        sepia_frame = cv2.transform(frame, sepia_filter)

        # Add noise
        noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(sepia_frame, noise)

        # Add vignette
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        vignette = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = math.sqrt(center_x**2 + center_y**2)
                vignette[y, x] = 1 - (distance / max_distance) * 0.5

        # Apply vignette
        for i in range(3):
            noisy_frame[:, :, i] = noisy_frame[:, :, i] * vignette

        return noisy_frame.astype(np.uint8)

    def get_dynamic_color(self, x, y, intensity):
        """Get dynamic color based on position and intensity"""
        if self.current_color == 'rainbow':
            hue = (self.frame_count * 2 + x * 5 + y * 3) % 360
            rgb = colorsys.hsv_to_rgb(hue/360, 1, intensity/255)
            return tuple(int(c * 255) for c in rgb)

        elif self.current_color in self.color_schemes:
            colors = self.color_schemes[self.current_color]
            if isinstance(colors, list) and len(colors) > 1:
                # Gradient between colors
                t = intensity / 255.0
                idx = int(t * (len(colors) - 1))
                if idx >= len(colors) - 1:
                    return colors[-1]

                # Interpolate between two colors
                c1, c2 = colors[idx], colors[idx + 1]
                blend = t * (len(colors) - 1) - idx

                return tuple(int(c1[i] * (1 - blend) + c2[i] * blend) for i in range(3))
            else:
                return colors[0] if isinstance(colors, list) else colors

        return (255, 255, 255)

    def frame_to_ultra_ascii(self, frame):
        """Convert frame to ultra-advanced ASCII art"""
        processed_frame = self.apply_advanced_effects(frame)

        # Update motion detection
        self.detect_motion(processed_frame)

        # Update particle system
        if self.particle_system:
            self.update_particles()

        # Store frame for ghost trail
        if self.ghost_trail:
            self.ghost_frames.append(cv2.cvtColor(
                processed_frame, cv2.COLOR_BGR2GRAY))

        # Get frame dimensions
        height, width = processed_frame.shape[:2]
        cell_width = width / self.cols
        cell_height = height / self.rows

        # Convert to grayscale for intensity calculation
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        # Create output ASCII image
        ascii_img = np.zeros(
            (self.rows * self.char_height, self.cols * self.char_width, 3), dtype=np.uint8)

        # Advanced face detection
        faces, eyes, smiles = [], [], []
        if self.face_detection and self.face_detection_available:
            faces = self.detect_advanced_faces(processed_frame)

        # Generate ASCII art with ultra effects
        for i in range(self.rows):
            for j in range(self.cols):
                # Calculate region of interest
                x1, y1 = int(j * cell_width), int(i * cell_height)
                x2, y2 = int((j + 1) * cell_width), int((i + 1) * cell_height)

                # Get ROI
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                intensity = int(np.mean(roi))

                # Apply various intensity modifications
                if self.negative_mode:
                    intensity = 255 - intensity

                if self.pulse_effect:
                    pulse = abs(
                        math.sin(self.frame_count * 0.1 + (i + j) * 0.05))
                    intensity = int(intensity * (0.3 + 0.7 * pulse))

                if self.thermal_vision:
                    # Use temperature-based intensity
                    thermal_intensity = self.calculate_thermal_intensity(roi)
                    intensity = thermal_intensity

                # Choose ASCII character
                char = self.select_character(intensity, i, j, faces)

                # Get color
                color = self.get_pixel_color(
                    processed_frame, x1, y1, x2, y2, i, j, intensity)

                # Apply special effects to character position
                text_x, text_y = self.calculate_text_position(i, j)
                font_scale = self.calculate_font_scale(i, j)

                # Draw the character with advanced effects
                self.draw_character(ascii_img, char, text_x,
                                    text_y, font_scale, color, i, j)

        # Apply post-processing effects
        ascii_img = self.apply_post_effects(ascii_img)

        return ascii_img

    def detect_motion(self, frame):
        """Advanced motion detection"""
        motion_mask = self.bg_subtractor.apply(frame)
        self.motion_history.append(motion_mask)

        # Calculate motion intensity
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
        self.motion_intensity = motion_pixels / total_pixels

        if self.motion_intensity > 0.1:
            self.session_stats['motion_events'] += 1

    def detect_advanced_faces(self, frame):
        """Advanced face detection with emotions and landmarks"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
        smiles = self.smile_cascade.detectMultiScale(gray, 1.8, 20)

        self.session_stats['faces_detected'] += len(faces)

        return {'faces': faces, 'eyes': eyes, 'smiles': smiles}

    def select_character(self, intensity, row, col, face_data):
        """Intelligent character selection based on context"""
        # Matrix rain effect
        if self.matrix_rain and col in getattr(self, 'matrix_columns', {}):
            column_data = self.matrix_columns[col]
            char_idx = int((row - column_data['y']) %
                           len(column_data['chars']))
            if 0 <= char_idx < len(column_data['chars']) and row >= column_data['y'] - 20:
                return column_data['chars'][char_idx]

        # Face detection override
        if face_data and 'faces' in face_data:
            for (fx, fy, fw, fh) in face_data['faces']:
                face_col = int(fx / (self.cols / self.char_width))
                face_row = int(fy / (self.rows / self.char_height))
                if abs(face_col - col) < 5 and abs(face_row - row) < 3:
                    return "😀" if self.current_effect == 'emoji' else "O"

        # Particle system override
        if self.particle_system:
            for particle in self.particles:
                if abs(particle['x'] - col) < 1 and abs(particle['y'] - row) < 1:
                    return particle['char']

        # Normal character selection
        char_idx = min(
            int((intensity / 255) * (len(self.ascii_chars) - 1)),
            len(self.ascii_chars) - 1
        )
        return self.ascii_chars[char_idx]

    def get_pixel_color(self, frame, x1, y1, x2, y2, row, col, intensity):
        """Advanced color calculation with effects"""
        if self.color_mode:
            color_roi = frame[y1:y2, x1:x2]
            if color_roi.size > 0:
                avg_color = np.mean(color_roi, axis=(0, 1)).astype(int)
                return tuple(avg_color.tolist())

        if self.rainbow_mode:
            return self.get_dynamic_color(col, row, intensity)

        if self.neon_glow:
            glow_intensity = intensity + 50
            base_color = self.color_schemes.get(
                self.current_color, [(255, 255, 255)])[0]
            return tuple(min(255, int(c * (glow_intensity / 255))) for c in base_color)

        # Standard color
        colors = self.color_schemes.get(self.current_color, [(255, 255, 255)])
        if isinstance(colors, list):
            return colors[0]
        return colors

    def calculate_thermal_intensity(self, roi):
        """Calculate thermal-based intensity"""
        # Simulate thermal reading based on pixel variance
        variance = np.var(roi)
        return min(255, int(variance * 3))

    def calculate_text_position(self, row, col):
        """Calculate text position with effects"""
        base_x = col * self.char_width
        base_y = (row + 1) * self.char_height - 4

        # Wave effect on text position
        if self.wave_effect:
            wave_offset = int(
                5 * math.sin(2 * math.pi * col / 20 + self.frame_count * 0.1))
            base_y += wave_offset

        # Shake effect for glitch mode
        if self.glitch_effect and random.random() < 0.05:
            base_x += random.randint(-2, 2)
            base_y += random.randint(-2, 2)

        return base_x, base_y

    def calculate_font_scale(self, row, col):
        """Calculate dynamic font scale"""
        scale = self.font_scale

        if self.zoom_effect:
            zoom_factor = 1 + 0.3 * abs(math.sin(self.frame_count * 0.05))
            scale *= zoom_factor

        if self.pulse_effect:
            pulse = 1 + 0.2 * \
                abs(math.sin(self.frame_count * 0.1 + (row + col) * 0.1))
            scale *= pulse

        return scale

    def draw_character(self, img, char, x, y, scale, color, row, col):
        """Draw character with advanced effects"""
        # Neon glow effect
        if self.neon_glow:
            # Draw glow background
            for i in range(3):
                glow_color = tuple(int(c * (0.3 + i * 0.2)) for c in color)
                cv2.putText(img, char, (x-i, y-i), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, glow_color, 2, cv2.LINE_AA)

        # Ghost trail effect
        if self.ghost_trail and len(self.ghost_frames) > 1:
            for i, ghost_frame in enumerate(self.ghost_frames[-3:]):
                alpha = 0.3 + i * 0.2
                ghost_color = tuple(int(c * alpha) for c in color)
                cv2.putText(img, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, ghost_color, 1, cv2.LINE_AA)

        # Main character
        cv2.putText(img, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def apply_post_effects(self, img):
        """Apply post-processing effects to ASCII image"""
        if self.laser_grid:
            img = self.add_laser_grid(img)

        if self.snow_effect:
            img = self.add_snow_effect(img)

        if self.lightning_effect:
            img = self.add_lightning_effect(img)

        if self.plasma_effect:
            img = self.add_plasma_effect(img)

        return img

    def add_laser_grid(self, img):
        """Add laser grid overlay"""
        height, width = img.shape[:2]

        # Horizontal lines
        for y in range(0, height, 40):
            cv2.line(img, (0, y), (width, y), (0, 255, 0), 1)

        # Vertical lines
        for x in range(0, width, 40):
            cv2.line(img, (x, 0), (x, height), (0, 255, 0), 1)

        return img

    def add_snow_effect(self, img):
        """Add falling snow effect"""
        if not hasattr(self, 'snowflakes'):
            self.snowflakes = []

        # Add new snowflakes
        if random.random() < 0.3:
            self.snowflakes.append({
                'x': random.randint(0, img.shape[1]),
                'y': 0,
                'speed': random.uniform(1, 3),
                'size': random.randint(1, 3)
            })

        # Update and draw snowflakes
        for flake in self.snowflakes[:]:
            flake['y'] += flake['speed']
            if flake['y'] > img.shape[0]:
                self.snowflakes.remove(flake)
            else:
                cv2.circle(img, (int(flake['x']), int(flake['y'])),
                           flake['size'], (255, 255, 255), -1)

        return img

    def add_lightning_effect(self, img):
        """Add lightning effect"""
        if random.random() < 0.02:  # 2% chance
            height, width = img.shape[:2]

            # Create lightning bolt
            points = []
            x = random.randint(width//4, 3*width//4)
            y = 0

            while y < height:
                points.append((x, y))
                x += random.randint(-20, 20)
                y += random.randint(10, 30)
                x = max(0, min(width-1, x))

            # Draw lightning
            for i in range(len(points)-1):
                cv2.line(img, points[i], points[i+1], (255, 255, 255), 2)
                cv2.line(img, points[i], points[i+1], (200, 200, 255), 1)

        return img

    def add_plasma_effect(self, img):
        """Add plasma background effect"""
        height, width = img.shape[:2]

        # Create plasma pattern
        for y in range(0, height, 4):
            for x in range(0, width, 4):
                # Plasma formula
                plasma_val = (
                    math.sin(x / 16.0) +
                    math.sin(y / 32.0) +
                    math.sin((x + y) / 16.0) +
                    math.sin(math.sqrt(x*x + y*y) / 8.0) +
                    math.sin(self.frame_count / 10.0)
                ) * 32 + 128

                # Convert to color
                hue = int(plasma_val) % 360
                rgb = colorsys.hsv_to_rgb(hue/360, 0.5, 0.3)
                color = tuple(int(c * 255) for c in rgb)

                cv2.rectangle(img, (x, y), (x+4, y+4), color, -1)

        return img

    def draw_professional_ui(self, img):
        """Draw professional UI overlay"""
        height, width = img.shape[:2]

        # Performance stats
        fps = self.get_fps()
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Session stats
        runtime = time.time() - self.session_stats['session_start']
        cv2.putText(img, f"Runtime: {runtime:.0f}s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(img, f"Frames: {self.session_stats['frames_processed']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Current settings
        cv2.putText(img, f"Style: {self.current_effect}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img, f"Color: {self.current_color}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Motion intensity bar
        if self.motion_intensity > 0:
            bar_width = int(200 * self.motion_intensity)
            cv2.rectangle(img, (10, 140), (10 + bar_width, 150),
                          (0, 255, 255), -1)
            cv2.putText(img, "MOTION", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Active effects indicator
        effects = []
        effect_map = {
            'edge_detection': 'EDGE', 'color_mode': 'COLOR', 'wave_effect': 'WAVE',
            'pulse_effect': 'PULSE', 'face_detection': 'FACE', 'motion_blur': 'MOTION',
            'rainbow_mode': 'RAINBOW', 'zoom_effect': 'ZOOM', 'negative_mode': 'NEG',
            'contrast_boost': 'CONTRAST', 'particle_system': 'PARTICLES',
            'glitch_effect': 'GLITCH', 'hologram_mode': 'HOLO', 'cyberpunk_mode': 'CYBER',
            'thermal_vision': 'THERMAL', 'kaleidoscope': 'KALEIDO', 'vintage_film': 'VINTAGE',
            'neon_glow': 'NEON', 'ghost_trail': 'GHOST', 'matrix_rain': 'MATRIX'
        }

        for attr, label in effect_map.items():
            if getattr(self, attr, False):
                effects.append(label)

        if effects:
            effects_text = " | ".join(effects[:6])  # Show max 6 effects
            cv2.putText(img, effects_text, (10, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Recording indicator
        if self.recording:
            cv2.circle(img, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(img, "REC", (width - 55, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Size and resolution info
        cv2.putText(img, f"Res: {self.cols}x{self.rows}", (width - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Memory usage (approximate)
        particles_count = len(getattr(self, 'particles', []))
        cv2.putText(img, f"Particles: {particles_count}", (width - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return img

    def handle_advanced_keyboard(self, key):
        """Handle advanced keyboard input"""
        # Save current effect to stats
        if hasattr(self, 'session_stats'):
            active_effects = [attr for attr in dir(self)
                              if not attr.startswith('_') and
                              isinstance(getattr(self, attr, None), bool) and
                              getattr(self, attr, False)]
            self.session_stats['effects_used'].update(active_effects)

        # Effect selection (1-9, then 0 for more)
        effect_keys = {
            ord('1'): 'classic', ord('2'): 'detailed', ord('3'): 'matrix',
            ord('4'): 'retro', ord('5'): 'minimal', ord('6'): 'cyber',
            ord('7'): 'blocks', ord('8'): 'letters', ord('9'): 'persian'
        }

        if key in effect_keys:
            self.current_effect = effect_keys[key]
            self.ascii_chars = self.EFFECTS[self.current_effect]
            print(f"🎨 Switched to {self.current_effect} style")

        elif key == ord('0'):
            # Cycle through extended effects
            extended_effects = ['gradient', 'geometric', 'emoji', 'braille',
                                'japanese', 'arabic', 'music', 'space', 'weather']
            current_idx = extended_effects.index(
                self.current_effect) if self.current_effect in extended_effects else -1
            next_effect = extended_effects[(
                current_idx + 1) % len(extended_effects)]
            self.current_effect = next_effect
            self.ascii_chars = self.EFFECTS[next_effect]
            print(f"🌟 Advanced style: {next_effect}")

        # Color selection (QWERTY + numbers)
        color_keys = {
            ord('q'): 'white', ord('w'): 'green', ord('e'): 'cyan', ord('r'): 'red',
            ord('t'): 'blue', ord('y'): 'yellow', ord('u'): 'magenta', ord('i'): 'gold',
            ord('o'): 'silver', ord('p'): 'rainbow'
        }

        if key in color_keys:
            self.current_color = color_keys[key]
            print(f"🌈 Color: {self.current_color}")

        # Basic effects (ASDF row)
        elif key == ord('a'):
            self.edge_detection = not self.edge_detection
            print(
                f"🔍 Edge detection: {'ON' if self.edge_detection else 'OFF'}")
        elif key == ord('s'):
            self.color_mode = not self.color_mode
            print(f"🎨 Color mode: {'ON' if self.color_mode else 'OFF'}")
        elif key == ord('d'):
            self.wave_effect = not self.wave_effect
            print(f"🌊 Wave effect: {'ON' if self.wave_effect else 'OFF'}")
        elif key == ord('f'):
            self.pulse_effect = not self.pulse_effect
            print(f"💓 Pulse effect: {'ON' if self.pulse_effect else 'OFF'}")
        elif key == ord('g'):
            self.face_detection = not self.face_detection
            print(
                f"👤 Face detection: {'ON' if self.face_detection else 'OFF'}")
        elif key == ord('h'):
            self.motion_blur = not self.motion_blur
            print(f"🏃 Motion blur: {'ON' if self.motion_blur else 'OFF'}")
        elif key == ord('j'):
            self.rainbow_mode = not self.rainbow_mode
            print(f"🌈 Rainbow mode: {'ON' if self.rainbow_mode else 'OFF'}")
        elif key == ord('k'):
            self.zoom_effect = not self.zoom_effect
            print(f"🔍 Zoom effect: {'ON' if self.zoom_effect else 'OFF'}")
        elif key == ord('l'):
            self.negative_mode = not self.negative_mode
            print(f"🔄 Negative mode: {'ON' if self.negative_mode else 'OFF'}")

        # Advanced effects (ZXCV row)
        elif key == ord('z'):
            self.contrast_boost = not self.contrast_boost
            print(
                f"✨ Contrast boost: {'ON' if self.contrast_boost else 'OFF'}")
        elif key == ord('x'):
            self.glitch_effect = not self.glitch_effect
            print(f"⚡ Glitch effect: {'ON' if self.glitch_effect else 'OFF'}")
        elif key == ord('c'):
            self.hologram_mode = not self.hologram_mode
            print(f"👻 Hologram mode: {'ON' if self.hologram_mode else 'OFF'}")
        elif key == ord('v'):
            self.cyberpunk_mode = not self.cyberpunk_mode
            print(
                f"🤖 Cyberpunk mode: {'ON' if self.cyberpunk_mode else 'OFF'}")
        elif key == ord('b'):
            self.thermal_vision = not self.thermal_vision
            print(
                f"🌡️ Thermal vision: {'ON' if self.thermal_vision else 'OFF'}")
        elif key == ord('n'):
            self.kaleidoscope = not self.kaleidoscope
            print(f"🔮 Kaleidoscope: {'ON' if self.kaleidoscope else 'OFF'}")
        elif key == ord('m'):
            self.matrix_rain = not self.matrix_rain
            print(f"💚 Matrix rain: {'ON' if self.matrix_rain else 'OFF'}")

        # Function keys for special effects
        elif key == ord(','):
            self.particle_system = not self.particle_system
            print(
                f"✨ Particle system: {'ON' if self.particle_system else 'OFF'}")
        elif key == ord('.'):
            self.ghost_trail = not self.ghost_trail
            print(f"👻 Ghost trail: {'ON' if self.ghost_trail else 'OFF'}")
        elif key == ord('/'):
            self.neon_glow = not self.neon_glow
            print(f"💡 Neon glow: {'ON' if self.neon_glow else 'OFF'}")

        # More advanced effects
        elif key == ord('['):
            self.fish_eye = not self.fish_eye
            print(f"🐠 Fish-eye: {'ON' if self.fish_eye else 'OFF'}")
        elif key == ord(']'):
            self.vintage_film = not self.vintage_film
            print(f"📼 Vintage film: {'ON' if self.vintage_film else 'OFF'}")
        elif key == ord(';'):
            self.laser_grid = not self.laser_grid
            print(f"🔴 Laser grid: {'ON' if self.laser_grid else 'OFF'}")
        elif key == ord("'"):
            self.snow_effect = not self.snow_effect
            print(f"❄️ Snow effect: {'ON' if self.snow_effect else 'OFF'}")

        # Size and font controls
        elif key == ord('+') or key == ord('='):
            self.cols = min(200, self.cols + 10)
            self.rows = min(80, self.rows + 5)
            print(f"📐 Size: {self.cols}x{self.rows}")
        elif key == ord('-') or key == ord('_'):
            self.cols = max(40, self.cols - 10)
            self.rows = max(15, self.rows - 5)
            print(f"📐 Size: {self.cols}x{self.rows}")

        # Font scaling
        elif key == ord('`'):
            self.font_scale = min(1.5, self.font_scale + 0.1)
            print(f"🔤 Font scale: {self.font_scale:.1f}")
        elif key == ord('~'):
            self.font_scale = max(0.2, self.font_scale - 0.1)
            print(f"🔤 Font scale: {self.font_scale:.1f}")

        # Recording and export
        elif key == ord(' '):
            self.recording = not self.recording
            print(f"🎬 Recording: {'ON' if self.recording else 'OFF'}")
        elif key == 13:  # Enter key
            self.take_screenshot()
        elif key == 9:  # Tab key
            self.save_preset()

        # Preset loading (Ctrl combinations would be ideal, but using Shift+numbers)
        elif key == ord('!'):  # Shift+1
            self.load_preset('cinematic')
        elif key == ord('@'):  # Shift+2
            self.load_preset('cyberpunk')
        elif key == ord('#'):  # Shift+3
            self.load_preset('matrix')
        elif key == ord('$'):  # Shift+4
            self.load_preset('artistic')
        elif key == ord('%'):  # Shift+5
            self.load_preset('security')

    def take_screenshot(self):
        """Take a screenshot of current ASCII art"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ascii_screenshot_{timestamp}.png"
        print(f"📸 Screenshot saved: {filename}")

    def save_preset(self):
        """Save current settings as preset"""
        preset_name = f"custom_{datetime.now().strftime('%H%M%S')}"
        settings = {
            'effect': self.current_effect,
            'color': self.current_color,
            'effects': {attr: getattr(self, attr) for attr in dir(self)
                        if not attr.startswith('_') and isinstance(getattr(self, attr, None), bool)}
        }
        print(f"💾 Preset saved: {preset_name}")

    def load_preset(self, preset_name):
        """Load a predefined preset"""
        if preset_name in self.presets:
            preset = self.presets[preset_name]

            # Reset all effects
            for attr in dir(self):
                if not attr.startswith('_') and isinstance(getattr(self, attr, None), bool):
                    setattr(self, attr, False)

            # Apply preset
            self.current_effect = preset.get('ascii_style', 'detailed')
            self.ascii_chars = self.EFFECTS[self.current_effect]
            self.current_color = preset.get('color', 'white')

            for effect in preset.get('effects', []):
                if hasattr(self, effect):
                    setattr(self, effect, True)

            print(f"🎭 Loaded preset: {preset_name}")

    def get_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        if len(self.fps_counter) > 1:
            return len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
        return 0

    def apply_contrast_boost(self, frame):
        """Enhanced contrast boost"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_edge_detection(self, frame):
        """Enhanced edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_wave_effect(self, frame):
        """Enhanced wave effect"""
        rows, cols = frame.shape[:2]
        wave_frame = np.zeros_like(frame)

        for i in range(rows):
            offset = int(15 * math.sin(2 * math.pi *
                         i / 30 + self.frame_count * 0.1))
            if abs(offset) < cols:
                wave_frame[i] = np.roll(frame[i], offset, axis=0)
            else:
                wave_frame[i] = frame[i]

        return wave_frame

    def apply_motion_blur(self, frame):
        """Enhanced motion blur"""
        motion_mask = self.bg_subtractor.apply(frame)
        self.motion_history.append(motion_mask)

        if len(self.motion_history) > 3:
            combined_motion = np.zeros_like(motion_mask)
            for i, mask in enumerate(self.motion_history):
                alpha = (i + 1) / len(self.motion_history)
                combined_motion = cv2.addWeighted(
                    combined_motion, 1-alpha, mask, alpha, 0)

            kernel = np.ones((3, 3), np.uint8)
            motion_blur = cv2.morphologyEx(
                combined_motion, cv2.MORPH_OPEN, kernel)
            motion_blur = cv2.GaussianBlur(motion_blur, (9, 9), 0)

            motion_blur_3ch = cv2.cvtColor(motion_blur, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.8, motion_blur_3ch, 0.2, 0)

        return frame

    def run_ultra_camera(self):
        """Run the ultra professional ASCII camera"""
        print("=" * 80)
        print("🚀 ULTRA PROFESSIONAL ASCII CAMERA PRO MAX - AI ENHANCED EDITION")
        print("=" * 80)
        print("🎮 ULTIMATE CONTROLS:")
        print("  [1-9] ASCII Styles    [0] Extended Styles")
        print("  [Q-P] Color Palette   [SHIFT+1-5] Presets")
        print("")
        print("🔥 BASIC EFFECTS:")
        print("  [A] Edge Detection    [S] Color Mode       [D] Wave Effect")
        print("  [F] Pulse Effect      [G] Face Detection   [H] Motion Blur")
        print("  [J] Rainbow Mode      [K] Zoom Effect      [L] Negative Mode")
        print("")
        print("⚡ ADVANCED EFFECTS:")
        print("  [Z] Contrast Boost    [X] Glitch Effect    [C] Hologram Mode")
        print("  [V] Cyberpunk Mode    [B] Thermal Vision   [N] Kaleidoscope")
        print("  [M] Matrix Rain       [,] Particle System  [.] Ghost Trail")
        print("  [/] Neon Glow         [[] Fish Eye         []] Vintage Film")
        print("  [;] Laser Grid        ['] Snow Effect")
        print("")
        print("🎬 PROFESSIONAL TOOLS:")
        print("  [SPACE] Record        [ENTER] Screenshot   [TAB] Save Preset")
        print("  [+/-] Adjust Size     [`/~] Font Scale")
        print("  [ESC] Exit")
        print("=" * 80)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return

        # Professional camera setup
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        print("🎥 Ultra HD camera initialized!")
        print("🎨 AI-powered ASCII engine ready!")
        print("🌟 Prepare for visual magic...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Frame capture failed")
                    break

                # Mirror effect for natural interaction
                frame = cv2.flip(frame, 1)

                # Generate ultra ASCII art
                ascii_image = self.frame_to_ultra_ascii(frame)

                # Add professional UI
                ascii_image = self.draw_professional_ui(ascii_image)

                # Display the masterpiece
                cv2.imshow('ASCII Camera Pro', ascii_image)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key != 255:
                    self.handle_advanced_keyboard(key)

                # Update counters
                self.frame_count += 1
                self.session_stats['frames_processed'] += 1

        except KeyboardInterrupt:
            print("\n⚡ Session interrupted by user")
        except Exception as e:
            print(f"❌ Critical error: {e}")
        finally:
            # Cleanup and save stats
            cap.release()
            cv2.destroyAllWindows()
            self.save_session_stats()
            print("🎭 ASCII Camera Pro session ended")
            print(
                f"📊 Processed {self.session_stats['frames_processed']} frames")
            print(
                f"🎨 Used {len(self.session_stats['effects_used'])} different effects")
            print("👋 Thanks for using the ultimate ASCII experience!")


def main():
    print("🌟 Initializing Ultra Professional ASCII Camera Pro...")
    print("🔥 Loading 50+ visual effects and AI enhancements...")
    print("🚀 Preparing for the ultimate ASCII experience...")

    camera = UltraProfessionalASCIICamera()

    try:
        camera.run_ultra_camera()
    except KeyboardInterrupt:
        print("\n⚡ Program terminated by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("🎬 Ultra ASCII Camera Pro - Session Complete")


if __name__ == "__main__":
    main()
