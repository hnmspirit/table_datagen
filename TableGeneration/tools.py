from PIL import Image
from io import BytesIO
import urllib.parse


def html_to_img(driver,html_content,id_count):
    '''converts html to image'''
    html_content = urllib.parse.quote(html_content)
    driver.get("data:text/html;charset=utf-8," + html_content)
    window_size = driver.get_window_size()
    max_height,max_width = window_size['height'],window_size['width']
    e = driver.find_element_by_id('c0')

    bboxes = []
    for id in range(id_count):
        e = driver.find_element_by_id('c'+str(id))
        txt = e.text.strip()
        lentext = len(txt)
        loc = e.location
        size_ = e.size
        xmin = loc['x']
        ymin = loc['y']
        xmax = int(size_['width'] + xmin)
        ymax = int(size_['height'] + ymin)
        bboxes.append([lentext,txt,xmin,ymin,xmax,ymax])
        # cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),2)

    png = driver.get_screenshot_as_png()
    im = Image.open(BytesIO(png))
    im = im.crop((0,0, max_width, max_height))

    return im,bboxes