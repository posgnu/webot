import pyautogui

import time

while True:
    pyautogui.click(100, 100)
    time.sleep(3)


myScreenshot = pyautogui.screenshot()
myScreenshot.save("./example.png")
