[app]

# (str) Title of your application
title = My AR Project

# (str) Package name
package.name = myarapp

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py live
source.dir = .

# (str) Application versioning (add this line)
version = 0.1

# (list) Source code where the main.py live
source.include_exts = py,png,jpg,kv,atlas,mp4,mp3

# (list) Application requirements
# Note: 'opencv' is the recipe name for OpenCV
requirements = python3,kivy,numpy,opencv,ffpyplayer

# (list) Include specific files
source.include_files = test.mp4,test.mp3,template.png

# (list) Permissions
android.permissions = CAMERA,INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# (int) Target Android API
android.api = 33

# (int) Minimum API required
android.minapi = 21