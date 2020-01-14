#this is not my code, this is code from Sentdex's tutorial on python plays GTA V
#this will be stuff like reading the screen and direct input to games not really anything a.i.
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import pyautogui
import time

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def halfsecond():
    for i in list(range(5))[::-1]:
        time.sleep(0.1)
def keypress(output):
    k = np.argmax(output)
    if k == 0:
        k = 0
    if k == 1:
        pyautogui.press('up')
    if k == 2:
        pyautogui.press('down')
    if k == 3:
        pyautogui.press('left')
    if k == 4:
        pyautogui.press('right')
    if k == 5:
        pyautogui.press('1')
    if k == 6:
        pyautogui.press('2')
    if k == 7:
        pyautogui.press('3')
    if k == 8:
        pyautogui.press('4')
    if k == 9:
        pyautogui.press('5')
    if k == 10:
        pyautogui.press('6')
    if k == 11:
        pyautogui.press('7')
    if k == 12:
        pyautogui.press('8')
    if k == 13:
        pyautogui.press('9')
