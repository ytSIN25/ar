from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import pygame
import os

# 1. Define the UI Layout in Kivy Language (KV)
KV = '''
<MainLayout>:
    orientation: 'vertical'
    Image:
        id: camera_feed
        allow_stretch: True
'''

Builder.load_string(KV)

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. Request permissions on Android
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.READ_EXTERNAL_STORAGE])
        
        # Hardware/Files
        self.webcamCapture = cv2.VideoCapture(0)
        video_path = os.path.join(os.path.dirname(__file__), 'test.mp4')
        self.overlayVideo = cv2.VideoCapture(video_path)
        
        # Audio
        pygame.mixer.init()
        audio_path = os.path.join(os.path.dirname(__file__), 'test.mp3')
        pygame.mixer.music.load(audio_path)
        self.isAudioPlaying = False
        
        # Vision Setup
        template_path = os.path.join(os.path.dirname(__file__), 'template.png')
        self.templateImage = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.orbDetector = cv2.ORB_create(nfeatures=5000)
        self.templateKP, self.templateDesc = self.orbDetector.detectAndCompute(self.templateImage, None)
        self.featureMatcher = cv2.BFMatcher()
        
        # Tracking State
        self.confidenceFrameCount = 0
        self.MAX_CONFIDENCE = 10
        self.lastValidPoints = None
        self.previousDestinationPoints = None
        self.smoothingAlpha = 0.2
        
        Clock.schedule_once(self.start_loop)

    def request_android_permissions(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.READ_EXTERNAL_STORAGE])
    
    def start_loop(self, dt):
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        success, webcamFrame = self.webcamCapture.read()
        if not success: return

        # 1. Detection
        grayFrame = cv2.cvtColor(webcamFrame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orbDetector.detectAndCompute(grayFrame, None)
        
        homographyMatrix = None
        videoReadSuccess = False
        videoFrame = None

        if desc is not None and len(kp) > 30:
            matches = self.featureMatcher.knnMatch(self.templateDesc, desc, k=2)
            good = [m for m, n in matches if m.distance < 0.70 * n.distance]
            
            if len(good) > 40:
                self.confidenceFrameCount = min(self.confidenceFrameCount + 1, self.MAX_CONFIDENCE)
                src = np.float32([self.templateKP[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                homographyMatrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            else:
                self.confidenceFrameCount = max(self.confidenceFrameCount - 1, 0)

        # 2. Video & Audio
        if self.confidenceFrameCount > 0:
            videoReadSuccess, videoFrame = self.overlayVideo.read()
            if not videoReadSuccess:
                self.overlayVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
                videoReadSuccess, videoFrame = self.overlayVideo.read()
        
        # 3. Audio Control
        if self.confidenceFrameCount > 3:
            # Check if we should start or restart the music
            if not self.isAudioPlaying or not pygame.mixer.music.get_busy():
                video_msec = self.overlayVideo.get(cv2.CAP_PROP_POS_MSEC)
                pygame.mixer.music.play(start=video_msec / 1000.0)
                self.isAudioPlaying = True
        else:
            if self.isAudioPlaying:
                pygame.mixer.music.stop()
                self.isAudioPlaying = False

        # 3. Warping
        if (homographyMatrix is not None or self.lastValidPoints is not None) and videoReadSuccess:
            h, w = self.templateImage.shape
            corners = np.float32([[0,0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            if homographyMatrix is not None:
                pts = cv2.perspectiveTransform(corners, homographyMatrix)
                self.previousDestinationPoints = pts if self.previousDestinationPoints is None else \
                    (self.smoothingAlpha * pts + (1 - self.smoothingAlpha) * self.previousDestinationPoints)
                self.lastValidPoints = self.previousDestinationPoints
            
            stableH, _ = cv2.findHomography(corners, self.lastValidPoints, cv2.RANSAC, 5.0)
            if stableH is not None:
                warped = cv2.warpPerspective(cv2.resize(videoFrame, (w, h)), stableH, (webcamFrame.shape[1], webcamFrame.shape[0]))
                mask = cv2.fillConvexPoly(np.zeros_like(webcamFrame), np.int32(self.lastValidPoints), (255, 255, 255))
                webcamFrame = cv2.bitwise_and(webcamFrame, cv2.bitwise_not(mask))
                webcamFrame = cv2.add(webcamFrame, warped)

        # 4. Kivy Update
        buf = cv2.flip(webcamFrame, 0).tobytes()
        texture = Texture.create(size=(webcamFrame.shape[1], webcamFrame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.camera_feed.texture = texture

    def on_stop(self):
        self.webcamCapture.release()
        self.overlayVideo.release()
        pygame.mixer.quit()

class MyARApp(App):
    def build(self):
        return MainLayout()

if __name__ == '__main__':
    MyARApp().run()