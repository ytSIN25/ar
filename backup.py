import cv2
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# --- AUDIO SETUP ---
pygame.mixer.init()
pygame.mixer.music.load('test.mp3') 
isAudioPlaying = False

# 1. Setup Template
templateImage = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
orbDetector = cv2.ORB_create(nfeatures=4000)
templateKeypoints, templateDescriptors = orbDetector.detectAndCompute(templateImage, None)

# 2. Setup Video Sources
webcamCapture = cv2.VideoCapture(0)
overlayVideo = cv2.VideoCapture('test.mp4')
featureMatcher = cv2.BFMatcher()

# Smoothing & Tracking Variables
previousDestinationPoints = None
smoothingAlpha = 0.2 
confidenceFrameCount = 0
MAX_CONFIDENCE = 10
lastValidPoints = None

while True:
    webcamReadSuccess, webcamFrame = webcamCapture.read()
    
    # Logic change: We only read the video frame if we are actually confident enough to show it
    # This helps keep the video "paused" visually and logically
    videoReadSuccess = False
    if confidenceFrameCount > 0:
        videoReadSuccess, videoFrame = overlayVideo.read()

    if not webcamReadSuccess: 
        break

    if confidenceFrameCount > 0 and not videoReadSuccess:
        overlayVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pygame.mixer.music.play() 
        videoReadSuccess, videoFrame = overlayVideo.read()

    # Detection & Feature Matching
    grayWebcamFrame = cv2.cvtColor(webcamFrame, cv2.COLOR_BGR2GRAY)
    frameKeypoints, frameDescriptors = orbDetector.detectAndCompute(grayWebcamFrame, None)
    
    homographyMatrix = None

    if frameDescriptors is not None and len(frameKeypoints) > 30:
        rawMatches = featureMatcher.knnMatch(templateDescriptors, frameDescriptors, k=2)
        goodMatches = [m for m, n in rawMatches if m.distance < 0.70 * n.distance]

        if len(goodMatches) > 40:
            confidenceFrameCount = min(confidenceFrameCount + 1, MAX_CONFIDENCE)
            sourcePoints = np.float32([templateKeypoints[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            destinationPoints = np.float32([frameKeypoints[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            homographyMatrix, _ = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
        else:
            confidenceFrameCount = max(confidenceFrameCount - 1, 0)

    # --- SYNCHRONIZED AUDIO CONTROL ---
    if confidenceFrameCount > 3: # Increased threshold to reduce flickering
        if not isAudioPlaying:
            # Get current video time in seconds
            currentTimeSeconds = overlayVideo.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Play and immediately seek to the video's current position
            pygame.mixer.music.play(start=currentTimeSeconds)
            isAudioPlaying = True
    else:
        if isAudioPlaying:
            pygame.mixer.music.stop() # Use stop instead of pause to allow 'start' seek later
            isAudioPlaying = False

    # Rendering AR Overlay
    if (homographyMatrix is not None or (lastValidPoints is not None and confidenceFrameCount > 0)) and videoReadSuccess:
        templateHeight, templateWidth = templateImage.shape
        templateCorners = np.float32([[0,0], [0, templateHeight-1], [templateWidth-1, templateHeight-1], [templateWidth-1, 0]]).reshape(-1,1,2)
        
        if homographyMatrix is not None:
            currentDestinationPoints = cv2.perspectiveTransform(templateCorners, homographyMatrix)
            if previousDestinationPoints is None: 
                previousDestinationPoints = currentDestinationPoints
            
            smoothedPoints = (smoothingAlpha * currentDestinationPoints + (1 - smoothingAlpha) * previousDestinationPoints)
            previousDestinationPoints = smoothedPoints
            lastValidPoints = smoothedPoints
        else:
            smoothedPoints = lastValidPoints

        stableHomography, _ = cv2.findHomography(templateCorners, smoothedPoints, cv2.RANSAC, 5.0)
        
        if stableHomography is not None:
            videoFrameResized = cv2.resize(videoFrame, (templateWidth, templateHeight))
            warpedVideoFrame = cv2.warpPerspective(videoFrameResized, stableHomography, (webcamFrame.shape[1], webcamFrame.shape[0]))
            
            mask = np.zeros(webcamFrame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(smoothedPoints), 255)
            
            webcamFrame = cv2.bitwise_and(webcamFrame, webcamFrame, mask=cv2.bitwise_not(mask))
            webcamFrame = cv2.add(webcamFrame, warpedVideoFrame)

    cv2.imshow('AR Video Overlay', webcamFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcamCapture.release()
overlayVideo.release()
pygame.mixer.quit()
cv2.destroyAllWindows()